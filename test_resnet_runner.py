"""Small correctness checks for export, device routing and dataset handling."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
from torchvision.models import resnet50
import resnet_benchmark as runner


class RunnerTests(unittest.TestCase):
    def test_quantizer_half_ties_and_adjacent_values(self):
        # A16 makes exact half-integer ties common enough to affect predictions.
        ties = torch.arange(-32767.5, 32768, 1)
        values = torch.cat((ties, torch.nextafter(ties, torch.full_like(ties, -float("inf"))),
                            torch.nextafter(ties, torch.full_like(ties, float("inf")))))
        expected = values.round()
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                self.assertTrue(torch.equal(runner.round_to_even(values.to(device)).cpu(), expected))
                activation = runner.Activation().to(device)
                activation.bound.fill_(127)
                x = torch.tensor([-200, -2.5, -1.5, -.5, .5, 1.5, 2.5, 200.], device=device)
                self.assertEqual(activation(x, 8).cpu().tolist(), [-127, -2, -2, 0, 0, 2, 2, 127])

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = runner.QuantResNet(resnet50(weights=None).eval()).eval()

    def test_csv_covers_actual_quantizers_and_stage_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.csv"
            # All four choices, including A16 and projection convolutions.
            rows = runner.export_policy_csv(self.model, [0, 1, 2, 3], path)
            with path.open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), len(rows))
        self.assertEqual(len({r["name"] for r in rows}), len(rows))
        weights = [r for r in rows if r["kind"] == "weight"]
        activations = [r for r in rows if r["kind"] == "activation_quantizer"]
        self.assertEqual(len(weights), sum(isinstance(m, runner.Affine) for m in self.model.modules()))
        self.assertEqual(len(activations), sum(isinstance(m, runner.Activation) for m in self.model.modules()))
        self.assertEqual((len(weights), len(activations)), (54, 50))
        self.assertEqual(sum(r["parameter_count"] for r in weights), sum(m.count for m in self.model.modules() if isinstance(m, runner.Affine)))
        by_name = {r["name"]: r for r in rows}
        for row in weights:
            mode = (8, 8) if row["stage"] == "fixed" else runner.MODES[int(row["stage"][-1]) - 1]
            self.assertEqual((row["weight_bits"], row["activation_bits"]), mode)
            self.assertEqual(by_name[row["input_quantizer"]]["quant_bits"], mode[1])
            self.assertEqual(row["bias_bits"], 32)
        self.assertEqual(by_name["layer4.0.downsample.0"]["input_quantizer"], "layer4.0.input_quantizer")
        self.assertEqual(by_name["layer4.0.residual_add"]["compute_dtype"], "float32")
        self.assertEqual(by_name["fc.output"]["quant_bits"], "")
        with self.assertRaises(ValueError):
            runner.export_policy_csv(self.model, [0, 4, 2, 3], Path("unused.csv"))

    def test_cuda_routing_and_synchronization(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "device_count", return_value=2), patch.object(torch.cuda, "set_device") as set_device:
            self.assertEqual(runner.select_device("auto"), torch.device("cuda:0"))
            set_device.assert_called_with(torch.device("cuda:0"))
            self.assertEqual(runner.select_device("cuda:1").index, 1)
            set_device.assert_called_with(torch.device("cuda:1"))
            with self.assertRaises(ValueError):
                runner.select_device("cuda:2")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA requested but unavailable"):
                runner.select_device("cuda")
        with patch.object(torch.cuda, "synchronize") as synchronize:
            runner.sync(torch.device("cuda:1"))
            synchronize.assert_called_once_with(torch.device("cuda:1"))

    def test_imagenet_split_and_original_label_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                for wnid in ("n02102040", "n01440764"):
                    directory = root / split / wnid
                    directory.mkdir(parents=True)
                    for i in range(4):
                        (directory / f"{i}.JPEG").touch()
            result = runner.imagenet_manifest(root, (2, 2, 2), 42)
            self.assertEqual(result, runner.imagenet_manifest(root, (2, 2, 2), 42))
            self.assertEqual([len(v) for v in result.values()], [2, 2, 2, 8])
            files = [r["file"] for rows in result.values() for r in rows]
            self.assertEqual(len(files), len(set(files)))
            for split, rows in result.items():
                for row in rows:
                    self.assertEqual(Path(row["file"]).parent.parent.name, "val" if split == "test" else "train")
                    self.assertEqual(row["label"], 0 if Path(row["file"]).parent.name == "n01440764" else 217)
            with self.assertRaisesRegex(ValueError, "Not enough"):
                runner.imagenet_manifest(root, (9, 1, 1), 42)

    def test_manifest_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            image = Path(tmp) / "a.JPEG"
            image.touch()
            row = {"file": str(image), "label": 0}
            path.write_text(json.dumps({"observer": [row], "profile": [row], "search": [row]}))
            with patch.object(runner, "MANIFEST", path), self.assertRaisesRegex(ValueError, "more than once"):
                runner.read_manifest()

    def test_evaluation_streams_batches_and_handles_tail(self):
        class FakeModel:
            def __call__(self, x, policy=None):
                logits = torch.zeros(len(x), 1000)
                logits[:, 0 if policy is None else 1] = 10
                return logits
        calls = []
        def fake_load(rows):
            calls.append(len(rows))
            return torch.zeros(len(rows), 3, 2, 2), torch.tensor([r["label"] for r in rows])
        rows = [{"file": f"{i}.JPEG", "label": i % 2} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "OUT", Path(tmp)), patch.object(runner, "BATCH_SIZE", 2), patch.object(runner, "load_images", side_effect=fake_load), patch.object(runner, "export_policy_csv"):
            result = runner.evaluate_test(FakeModel(), {"test": rows}, [0, 0, 0, 0], torch.device("cpu"))
            self.assertEqual(calls, [2, 2, 1])
            self.assertEqual(result["images"], 5)
            self.assertEqual(result["float_top1_correct"], 3)
            self.assertEqual(result["mixed_top1_correct"], 2)
            with (Path(tmp) / "evaluation_predictions.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 5)

    def test_saved_calibration_is_restored_and_checkpoint_is_checked(self):
        bounds = {name: float(m.bound) for name, m in self.model.named_modules() if isinstance(m, runner.Activation)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected_policy.json"
            path.write_text(json.dumps({"policy": [0, 1, 2, 3], "activation_bounds": {name: 2.0 for name in bounds}}))
            with patch.object(runner, "select_device", return_value=torch.device("cpu")), patch.object(runner, "calibrated_model", return_value=(self.model, {})), patch.object(runner, "OUT", Path(tmp)), patch.object(runner, "evaluate_test") as evaluate:
                try:
                    runner.evaluate_saved(path)
                    evaluate.assert_called_once_with(self.model, {}, (0, 1, 2, 3), torch.device("cpu"))
                    self.assertTrue(all(float(m.bound) == 2.0 for m in self.model.modules() if isinstance(m, runner.Activation)))
                finally:
                    for name, m in self.model.named_modules():
                        if name in bounds:
                            m.bound.fill_(bounds[name])
            checkpoint = Path(tmp) / "checkpoint.pth"
            checkpoint.write_bytes(b"different checkpoint")
            path.write_text(json.dumps({"policy": [0, 1, 2, 3], "checkpoint_sha256": "invalid"}))
            with patch.object(runner, "CHECKPOINT", checkpoint), self.assertRaisesRegex(ValueError, "checkpoint hash"):
                runner.evaluate_saved(path)

    def test_pareto_dominance_and_equal_coordinates(self):
        # A bigger model must improve accuracy to remain on the frontier.
        self.assertEqual(runner.pareto_indices([1, 2, 3, 4], [5, 4, 4, 6]), [0, 1])
        self.assertEqual(runner.pareto_indices([1, 1, 2], [3, 3, 3]), [0, 1])
        self.assertEqual(runner.pareto_indices([1, 2, 3], [5, 4, 3]), [0, 1, 2])

    def test_frontier_candidates_use_search_accuracy_then_kl(self):
        policies = np.zeros((5, 4), dtype=int)
        measured = np.zeros((5, 4, 3))
        for i, errors in enumerate([2, 1, 1, 0, 0]):
            measured[i, :errors, 2] = 1
        measured[:, :, 0] = np.array([.1, .4, .2, .3, .1])[:, None]
        # At size 10, choose policy 2 by KL; policy 4 is larger with equal accuracy.
        selected = runner.frontier_candidates(policies, np.array([10, 10, 10, 20, 30]), np.ones(5), measured)
        self.assertEqual(selected, [2, 3])


if __name__ == "__main__":
    unittest.main()
