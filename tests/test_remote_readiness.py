"""CPU-verifiable server contracts plus a real CUDA check enabled on GPU hosts."""
import copy
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from amp import fisher_router as fr, runner


class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.conv(x).relu().mean((2, 3)))


class RemoteReadinessTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_new_precision_api_does_not_mix_with_legacy_flags(self):
        matmul = SimpleNamespace(fp32_precision="tf32", allow_tf32=True)
        cudnn = SimpleNamespace(fp32_precision="tf32", conv=SimpleNamespace(fp32_precision="tf32"), allow_tf32=True)
        backends = SimpleNamespace(fp32_precision="tf32", cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn)
        with patch.object(torch, "backends", backends), patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "device_count", return_value=2), patch.object(torch.cuda, "set_device") as select, patch.dict(os.environ, {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0"}):
            self.assertEqual(runner.select_device("cuda:1"), torch.device("cuda:1"))
            select.assert_called_once_with(torch.device("cuda:1"))
        self.assertEqual((backends.fp32_precision, matmul.fp32_precision, cudnn.fp32_precision, cudnn.conv.fp32_precision), ("ieee",) * 4)
        self.assertTrue(matmul.allow_tf32 and cudnn.allow_tf32)
        self.assertFalse(cudnn.benchmark)
        self.assertTrue(cudnn.deterministic)

    def test_forced_tf32_is_rejected(self):
        with patch.dict(os.environ, {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}), self.assertRaisesRegex(ValueError, "TF32"):
            runner.select_device("cuda")

    def test_preflight_dispatch_and_kernel_failure(self):
        properties = SimpleNamespace(name="Mock GPU", major=8, minor=0, total_memory=16 * 1024**3)
        requested = []
        original_ones = torch.ones
        def cpu_ones(*args, **kwargs):
            requested.append(kwargs["device"])
            kwargs["device"] = "cpu"
            return original_ones(*args, **kwargs)
        with patch.object(torch.cuda, "get_device_properties", return_value=properties), patch.object(torch, "ones", side_effect=cpu_ones), patch.object(runner, "sync") as synchronize:
            result = runner.device_runtime("cuda:1", check=True)
            self.assertEqual(result["forward_backward_check"], "passed")
            self.assertEqual(result["compute_capability"], [8, 0])
            self.assertEqual(requested, [torch.device("cuda:1")] * 3)
            synchronize.assert_called_once_with(torch.device("cuda:1"))
            with patch.object(runner.F, "conv2d", side_effect=RuntimeError("no kernel image")), self.assertRaisesRegex(RuntimeError, "CUDA forward/backward preflight failed.*no kernel image"):
                runner.device_runtime("cuda:1", check=True)
        with patch.object(torch.cuda, "get_device_properties") as properties:
            self.assertEqual(runner.device_runtime("cpu", check=True)["device"], "cpu")
            properties.assert_not_called()

    def test_full_class_layout_retains_all_labels_and_disjoint_training_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = {str(i): [f"n{i:08}", f"class{i}"] for i in range(1000)}
            (root / "imagenet_class_index.json").write_text(json.dumps(index))
            for wnid, _ in index.values():
                for split in ("train", "val"):
                    directory = root / split / wnid
                    directory.mkdir(parents=True)
                    for i in range(3 if split == "train" else 1):
                        (directory / f"{i}.JPEG").touch()
            with patch.object(runner, "DATA", root):
                first = runner.imagenet_manifest(root, (1024, 128, 512), 101)
                second = runner.imagenet_manifest(root, (1024, 128, 512), 101)
            self.assertEqual(first, second)
            self.assertEqual({k: len(v) for k, v in first.items()}, {"observer": 1024, "profile": 128, "search": 512, "test": 1000})
            self.assertEqual({r["label"] for r in first["test"]}, set(range(1000)))
            paths = [r["file"] for rows in first.values() for r in rows]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertTrue(all(Path(r["file"]).parent.parent.name == "train" for split in ("observer", "profile", "search") for r in first[split]))

    def test_missing_and_renumbered_class_map_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "DATA", Path(temp)):
            with self.assertRaisesRegex(ValueError, "Copy benchmark_data"):
                runner.imagenet_manifest(Path(temp), (1, 1, 1), 1)
            (Path(temp) / "imagenet_class_index.json").write_text(json.dumps({"0": ["n02102040", "dog"]}))
            with self.assertRaisesRegex(ValueError, "1,000 class indices"):
                runner.imagenet_manifest(Path(temp), (1, 1, 1), 1)

    def test_bad_image_error_identifies_the_file(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / "broken.JPEG"
            bad.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "Cannot decode image.*broken.JPEG"):
                runner.load_images([{"file": str(bad), "label": 0}])

    def test_50000_validation_rows_stream_in_bounded_batches(self):
        class FakeModel:
            def __call__(self, x, policy=None):
                logits = torch.zeros(len(x), 3)
                logits[:, 0 if policy is None else 1] = 4
                return logits
        calls = []
        def load(rows):
            calls.append(len(rows))
            return torch.zeros(len(rows), 1), torch.tensor([r["label"] for r in rows])
        rows = [{"file": f"val/{i}.JPEG", "label": i % 3} for i in range(50000)]
        fake = SimpleNamespace(BATCH_SIZE=257, load_images=load, sync=lambda _: None, log=lambda _: None)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.csv"
            stats, _ = fr.heldout_evaluate(fake, FakeModel(), rows, {"P01": (0,)}, torch.device("cpu"), path)
            with path.open() as stream:
                self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 50000)
        self.assertEqual(sum(calls), 50000)
        self.assertEqual(max(calls), 257)
        self.assertEqual(calls[-1], 50000 % 257)
        self.assertEqual(stats["FP32"]["correct"], 16667)
        self.assertEqual(stats["P01"]["correct"], 16667)
        self.assertEqual(stats["P01"]["images"], 50000)

    def test_nonfinite_heldout_logits_fail_instead_of_reporting_accuracy(self):
        fake = SimpleNamespace(BATCH_SIZE=2, load_images=lambda rows: (torch.zeros(len(rows), 1), torch.zeros(len(rows), dtype=torch.long)), sync=lambda _: None, log=lambda _: None)
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "Non-finite held-out logits.*FP32"):
            fr.heldout_evaluate(fake, lambda x: torch.full((len(x), 3), float("nan")), [{"file": "bad"}], {}, "cpu", Path(temp) / "partial.csv")

    def test_profiling_microbatch_is_configurable(self):
        torch.manual_seed(11)
        model = fr.LayerModel(ConvClassifier()).eval()
        calls = []
        fake = SimpleNamespace(PROFILE_BATCH_SIZE=1, BATCH_SIZE=2, sync=lambda _: None, log=lambda _: None,
                               load_images=lambda rows: (torch.ones(len(rows), 3, 4, 4), torch.zeros(len(rows), dtype=torch.long)))
        manifest = {"observer": [{"file": "x"}] * 2, "profile": [{"file": "y"}] * 3}
        original = fr.fisher_scores
        def scores(model, batches, device):
            batches = list(batches)
            calls.extend(len(x) for x, _ in batches)
            return original(model, batches, device)
        with tempfile.TemporaryDirectory() as temp, patch.object(fr, "profile_metadata", return_value={}), patch.object(fr, "fisher_scores", side_effect=scores):
            fr.prepare_profile(fake, model, manifest, "cpu", "hash", Path(temp) / "profile.json")
        self.assertEqual(calls, [1, 1, 1])


@unittest.skipUnless(torch.cuda.is_available(), "Requires a real CUDA GPU; automatically runs on the server")
class CUDAIntegrationTests(unittest.TestCase):
    def test_cuda_fisher_costs_and_policy_replay_against_cpu(self):
        torch.manual_seed(19)
        device = runner.select_device("cuda")
        runner.device_runtime(device, check=True)
        network = ConvClassifier().eval()
        cpu = fr.LayerModel(copy.deepcopy(network)).eval()
        gpu = fr.LayerModel(copy.deepcopy(network)).eval().to(device)
        x, y = torch.randn(5, 3, 8, 8), torch.tensor([0, 1, 2, 0, 1])
        fr.calibrate(cpu, [(x, y)], "cpu")
        gpu.restore_bounds(cpu.bounds())
        batches = [(x[:3], y[:3]), (x[3:], y[3:])]
        expected = fr.fisher_scores(cpu, batches, "cpu")
        actual = fr.fisher_scores(gpu, batches, device)
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=1e-7)
        np.testing.assert_array_equal(fr.cost_table(gpu), fr.cost_table(cpu))
        policy = (0, 3)
        restored = fr.restore_policy(gpu, fr.policy_document(cpu, policy, "same"), "same")
        torch.testing.assert_close(gpu(x.to(device), restored).cpu(), cpu(x, policy), rtol=2e-4, atol=2e-5)
        self.assertTrue(all(p.grad is None for p in gpu.parameters()))


if __name__ == "__main__":
    unittest.main()
