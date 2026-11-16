"""Independent checks for the isolated v2 CLI, formats and repair search."""
import copy
import io
import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from amp import fisher_router as original
from amp import fisher_router_v2 as fr, runner_v2 as runner


ALL = ((4, 8), (4, 16), (8, 8), (8, 16))
TWO = ((4, 8), (8, 8))


def measurement(errors=5, kl=.1):
    return {"errors": errors, "mean_kl": kl, "images": 100, "complete": True}


def tables(modes, counts=(4, 1, 1, 1)):
    costs = np.array([[[count * w, count * w * a] for w, a in modes] for count in counts], dtype=float)
    scores = np.array([[10 / (w * a) for w, a in modes] for _ in counts])
    return scores, costs


class FormatTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.network = nn.Sequential(nn.Linear(5, 6), nn.ReLU(), nn.Linear(6, 3))
        self.model = fr.LayerModel(copy.deepcopy(self.network))
        self.x, self.y = torch.randn(7, 5), torch.tensor([0, 2, 1, 2, 0, 1, 2])
        fr.calibrate(self.model, [(self.x, self.y)], "cpu")

    def test_supported_subsets_reordering_and_single_mode(self):
        for modes in (ALL, TWO, tuple(reversed(ALL)), ((8, 16), (4, 8)), ((4, 16),)):
            with self.subTest(modes=modes), patch.object(fr, "MODES", modes):
                fr.validate_modes()
                scores = fr.fisher_scores(self.model, [(self.x, self.y)], "cpu")
                costs = fr.cost_table(self.model)
                self.assertEqual(scores.shape, (2, len(modes)))
                self.assertEqual(costs.shape, (2, len(modes), 2))
                budgets = fr.storage_budgets(costs, 8)
                if len({w for w, _ in modes}) == 1:
                    self.assertEqual(len(budgets), 1)
                for budget in budgets:
                    seed, report = fr.solve_seed(scores, costs, budget)
                    self.assertTrue(report["optimal"])
                    feasible = [p for p in itertools.product(range(len(modes)), repeat=2)
                                if fr.feasible(fr.policy_cost(costs, p), budget)]
                    expected = min(scores[np.arange(2), p].sum() for p in feasible)
                    self.assertAlmostEqual(scores[np.arange(2), seed].sum(), expected)
                reference = fr.w8a8_bitops(self.model)
                self.assertEqual(reference, sum(op.affine.macs * 64 for op in self.model.operators.values()))

    def test_invalid_modes_fail_clearly(self):
        for modes in ((), ((4, 8), (4, 8)), ((2, 8),), ((4, True),), ((4,),), None):
            with self.subTest(modes=modes), patch.object(fr, "MODES", modes):
                with self.assertRaisesRegex(ValueError, "MODES"):
                    fr.validate_modes()

    def test_fisher_matches_original_columns_and_preserves_model(self):
        old_model = original.LayerModel(copy.deepcopy(self.network))
        original.calibrate(old_model, [(self.x, self.y)], "cpu")
        expected = original.fisher_scores(old_model, [(self.x, self.y)], "cpu")
        saved = {name: value.clone() for name, value in self.model.state_dict().items()}
        modes = ((8, 8), (4, 8))
        with patch.object(fr, "MODES", modes):
            batches = [(self.x[i:i + 3], self.y[i:i + 3]) for i in range(0, 7, 3)]
            actual = fr.fisher_scores(self.model, batches, "cpu")
        np.testing.assert_allclose(actual, expected[:, [2, 0]], rtol=3e-4, atol=2e-10)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, saved[name]), name)
        self.assertTrue(all(not op._forward_hooks for op in self.model.operators.values()))

    def test_policy_reload_uses_bits_and_original_evaluator_agrees(self):
        with patch.object(fr, "MODES", TWO):
            policy = (0, 1)
            logits = self.model(self.x, policy).detach()
            document = fr.policy_document(self.model, policy, "checkpoint")
            with tempfile.TemporaryDirectory() as temp:
                fr.export_policy(self.model, policy, Path(temp) / "policy.csv")
        with patch.object(fr, "MODES", tuple(reversed(TWO))):
            restored = fr.restore_policy(self.model, document, "checkpoint")
            self.assertEqual(restored, (1, 0))
            torch.testing.assert_close(self.model(self.x, restored), logits, rtol=0, atol=0)
        old_model = original.LayerModel(copy.deepcopy(self.network))
        restored = original.restore_policy(old_model, document, "checkpoint")
        torch.testing.assert_close(old_model(self.x, restored), logits, rtol=0, atol=0)
        with patch.object(fr, "MODES", ((4, 8),)):
            with self.assertRaisesRegex(ValueError, "disabled"):
                fr.restore_policy(self.model, document, "checkpoint")

    def test_cache_projection_reordering_and_fingerprints_are_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            scores = np.arange(8, dtype=float).reshape(2, 4)
            metadata = {"checkpoint": "same", "modes": [list(m) for m in ALL]}
            saved = {"metadata": metadata, "scores": scores.tolist(), "costs": fr.cost_table(self.model).tolist(),
                     "activation_bounds": self.model.bounds(),
                     "operator_macs": {n: op.affine.macs for n, op in self.model.operators.items()}}
            path.write_text(json.dumps(saved))
            before = path.read_bytes()
            fake_runner = SimpleNamespace(log=lambda _: None)
            for modes in (TWO, tuple(reversed(ALL)), ((8, 16),)):
                with patch.object(fr, "MODES", modes), patch.object(fr, "profile_metadata", return_value=dict(metadata, modes=[list(m) for m in modes])):
                    result, costs, _, _, hit = fr.prepare_profile(fake_runner, self.model, {}, "cpu", "same", path)
                    self.assertTrue(hit)
                    indices = [ALL.index(m) for m in modes]
                    np.testing.assert_array_equal(result, scores[:, indices])
                    np.testing.assert_array_equal(costs, np.array(saved["costs"])[:, indices])
            self.assertEqual(path.read_bytes(), before)
            with patch.object(fr, "profile_metadata", return_value=dict(metadata, checkpoint="different")):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    fr.prepare_profile(fake_runner, self.model, {}, "cpu", "same", path)
            saved["metadata"]["modes"] = [list(m) for m in TWO]
            saved["scores"] = scores[:, [0, 2]].tolist()
            saved["costs"] = np.array(saved["costs"])[:, [0, 2]].tolist()
            path.write_text(json.dumps(saved))
            before = path.read_bytes()
            with patch.object(fr, "profile_metadata", return_value=metadata):
                with self.assertRaisesRegex(ValueError, "missing an enabled format"):
                    fr.prepare_profile(fake_runner, self.model, {}, "cpu", "same", path)
            self.assertEqual(path.read_bytes(), before)


class SearchTests(unittest.TestCase):
    def test_joint_repair_crosses_four_layer_barrier(self):
        with patch.object(fr, "MODES", TWO):
            scores, costs = tables(TWO)
            scores = np.array([[4., 0.], [1., 0.], [1., 0.], [1., 0.]])
            seed, winner = (1, 0, 0, 0), (0, 1, 1, 1)
            # Every alternative except the four-layer repair is worse than the seed.
            def evaluate(policy, cutoff):
                return measurement(2 if policy == winner else 5 if policy == seed else 9)
            cap = fr.policy_cost(costs, seed)
            solved, report = fr.solve_seed(scores, costs, cap)
            self.assertTrue(report["optimal"])
            self.assertEqual(solved, seed)
            local = fr.run_search(scores, costs, [cap], [seed], evaluate, "local", 32)
            repair = fr.run_search(scores, costs, [cap], [seed], evaluate, "transfer", 32)
            self.assertEqual(local["incumbents"], [list(seed)])
            self.assertEqual(repair["incumbents"], [list(winner)])
            self.assertEqual(repair["records"][1]["changed_layers"], [0, 1, 2, 3])
            self.assertGreater(scores[np.arange(4), winner].sum(), scores[np.arange(4), seed].sum())
            self.assertTrue(all(np.all(np.array(r["cost"]) <= cap) for r in repair["records"]))
            self.assertTrue(all(g["complete"] for g in repair["groups"]))

    def test_shared_resource_caps_round_robin_and_deterministic_partial_groups(self):
        with patch.object(fr, "MODES", TWO):
            scores, costs = tables(TWO, (3, 3, 1, 1, 1, 1))
            seeds = [(1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)]
            budgets = [[1000, 100000], [1001, 100001]]
            def evaluate(p, cutoff):
                return measurement(1 if p == seeds[1] else 2 if p == seeds[0] else 8)
            runs = [fr.run_search(scores, costs, budgets, seeds, evaluate, "transfer", 6) for _ in range(2)]
            self.assertEqual([r["policy"] for r in runs[0]["records"]], [r["policy"] for r in runs[1]["records"]])
            result = runs[0]
            self.assertEqual([r["target_budget"] for r in result["records"][2:]], [0, 1, 0, 1])
            self.assertEqual(result["candidate_count"], 6)
            self.assertEqual(len({tuple(r["policy"]) for r in result["records"]}), 6)
            self.assertTrue(any(not g["complete"] for g in result["groups"]))
            for mode in ("local", "transfer"):
                trial = fr.run_search(scores, costs, budgets, seeds, evaluate, mode, 12)
                for r in trial["records"]:
                    if r["target_budget"] is not None:
                        self.assertTrue(np.all(np.array(r["seed_resource_delta"]) <= 0))
                np.testing.assert_array_equal(trial["seed_resource_caps"], [fr.policy_cost(costs, p) for p in seeds])

    def test_group_parent_stays_fixed_until_queue_finishes(self):
        with patch.object(fr, "MODES", TWO):
            scores, costs = tables(TWO)
            seed = (1, 0, 0, 0)
            result = fr.run_search(scores, costs, [[1000]], [seed],
                                   lambda p, _: measurement(5 if p == seed else 2), "transfer", 8)
            group = result["groups"][0]
            first = [r for r in result["records"] if r["group_id"] == group["id"]]
            self.assertGreater(len(first), 1)
            self.assertTrue(all(r["parent"] == list(seed) for r in first))

    def test_frontier_race_keeps_intermediate_tradeoff(self):
        with patch.object(fr, "MODES", TWO):
            scores, costs = tables(TWO, (1, 1))
            seeds = [(0, 0), (1, 1)]
            def evaluate(p, cutoff):
                errors = {(0, 0): 6, (0, 1): 3, (1, 0): 3, (1, 1): 1}[p]
                if cutoff is not None and errors > cutoff:
                    return dict(measurement(cutoff + 1), images=5, complete=False)
                return measurement(errors)
            result = fr.run_search(scores, costs, [[8], [16]], seeds, evaluate, "local", 4)
            self.assertTrue(all(r["complete"] for r in result["records"]))
            front = fr.search_frontier_policies(result["records"], 0)
            self.assertTrue(any(sum(p) == 1 for p in front))

    def test_cutoff_requires_references_on_both_axes(self):
        records = [dict(measurement(2), cost=[20, 5]), dict(measurement(7), cost=[5, 20]),
                   dict(measurement(0), cost=[1, 1], complete=False)]
        self.assertEqual(fr.frontier_cutoff(records, [10, 10]), 7)
        self.assertIsNone(fr.frontier_cutoff(records, [4, 10]))
        self.assertIsNone(fr.frontier_cutoff([], [10, 10]))

    def test_ties_finish_for_kl_and_early_rejections_count(self):
        class IndexedModel:
            def __call__(self, x, policy=None):
                predictions = x[:, 0].long()
                return torch.nn.functional.one_hot(predictions, 2).float() * 3
        x = torch.tensor([[1.], [1.], [0.], [1.], [0.], [0.]])
        labels = torch.zeros(6, dtype=torch.long)
        teacher = torch.tensor([[.9, .1]]).repeat(6, 1)
        evaluate = fr.search_evaluator(IndexedModel(), x, labels, teacher, "cpu", 2, lambda _: None)
        self.assertFalse(evaluate((0,), 2)["complete"])
        self.assertTrue(evaluate((0,), 3)["complete"])
        with patch.object(fr, "MODES", TWO):
            scores, costs = tables(TWO)
            calls = []
            def rejected(p, cutoff):
                calls.append(p)
                return measurement(1) if cutoff is None else dict(measurement(2), complete=False)
            result = fr.run_search(scores, costs, [[1000]], [(1, 0, 0, 0)], rejected, "local", 4)
            self.assertEqual(result["candidate_count"], 4)
            self.assertEqual(len(calls), 4)
            self.assertTrue(any(not r["complete"] for r in result["records"]))

    def test_generic_groups_cover_activations_and_small_models(self):
        for modes in (ALL, ((4, 8), (4, 16)), tuple(reversed(ALL))):
            with self.subTest(modes=modes), patch.object(fr, "MODES", modes):
                scores, costs = tables(modes, (1, 1))
                most_expensive = max(range(len(modes)), key=lambda m: (modes[m][0] * modes[m][1], modes[m][0]))
                seed = (most_expensive,) * 2
                result = fr.run_search(scores, costs, [[1000]], [seed], lambda *_: measurement(), "transfer", 100)
                self.assertEqual(result["candidate_count"], len(modes) ** 2)
                self.assertTrue(all(g["complete"] for g in result["groups"]))
                self.assertEqual(result["stop_reason"], "no feasible untested proposal remains")

    def test_four_format_group_enumeration_preserves_other_layers(self):
        with patch.object(fr, "MODES", ALL):
            scores, costs = tables(ALL, (1,) * 6)
            parent = (3,) * 6
            candidates = fr.group_assignments(parent, (0, 1, 2, 3, 4), scores, costs,
                                             fr.policy_cost(costs, parent))
            self.assertEqual(len(candidates), 4 ** 5 - 1)
            self.assertEqual(len(set(candidates)), len(candidates))
            self.assertTrue(all(p[5] == parent[5] for p in candidates))

    def test_local_alternates_weight_and_activation_proposals(self):
        with patch.object(fr, "MODES", ALL):
            scores, costs = tables(ALL, (1, 1))
            seed = (3, 3)
            result = fr.run_search(scores, costs, [[1000]], [seed],
                                   lambda p, _: measurement(1 if p == seed else 9), "local", 3)
            first, second = [r["policy"] for r in result["records"][1:]]
            self.assertTrue(any(ALL[m][0] != 8 for m in first))
            self.assertTrue(all(ALL[m][0] == 8 for m in second))
            self.assertTrue(any(ALL[m][1] != 16 for m in second))

    def test_single_format_uniform_endpoints_duplicate_seeds_and_allowance(self):
        with patch.object(fr, "MODES", ((8, 16),)):
            scores, costs = tables(fr.MODES)
            result = fr.run_search(scores, costs, [[1000]] * 2, [(0,) * 4] * 2, lambda *_: measurement(), max_candidates=1)
            self.assertEqual(result["candidate_count"], 1)
            self.assertIn("single enabled format", result["stop_reason"])
        with patch.object(fr, "MODES", TWO):
            scores, costs = tables(TWO)
            seeds = [(0,) * 4, (1,) * 4]
            result = fr.run_search(scores, costs, [[28], [56]], seeds,
                                   lambda p, _: measurement(5 if p == seeds[0] else 1), max_candidates=32)
            self.assertEqual(result["candidate_count"], 2)
            self.assertEqual(result["groups"], [])
            with self.assertRaisesRegex(ValueError, "allowance"):
                fr.run_search(scores, costs, [[28], [56]], seeds, lambda *_: measurement(), max_candidates=1)


class IsolationTests(unittest.TestCase):
    def test_existing_output_is_rejected_before_logging(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "main.log"
            path.write_text("original log")
            with patch.object(runner, "OUT", runner.OUT), patch.object(runner, "run_logging") as logging, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    runner.main(["fisher-frontier", "--output-dir", temp])
            self.assertEqual(failure.exception.code, 2)
            logging.assert_not_called()
            self.assertEqual(path.read_text(), "original log")

    def test_json_publication_never_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            fr.write_json(path, {"original": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                fr.write_json(path, {"new": True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_v2_layer_dispatch_and_original_modules_are_separate(self):
        from amp import runner as old_runner
        self.assertIsNot(runner, old_runner)
        self.assertIsNot(fr, original)
        self.assertIsNot(fr.LayerModel, original.LayerModel)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "policy.json"
            path.write_text(json.dumps({"granularity": "operator"}))
            with patch.object(fr, "evaluate_saved") as evaluate:
                runner.evaluate_saved(path, export_only=True)
                evaluate.assert_called_once_with(runner, {"granularity": "operator"}, True)


class PipelineTests(unittest.TestCase):
    def test_selection_is_frozen_before_test_labels_and_unchanged_when_they_change(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        template = fr.LayerModel(nn.Sequential(nn.Linear(5, 6), nn.ReLU(), nn.Linear(6, 3)))
        images = torch.randn(8, 5)
        fr.calibrate(template, [(images, torch.zeros(8, dtype=torch.long))], "cpu")
        manifest = {split: [{"file": str(i), "label": i % 3} for i in indices]
                    for split, indices in {"observer": [0], "profile": [1], "search": [2, 3, 4], "test": [5, 6, 7]}.items()}
        frozen = []
        with tempfile.TemporaryDirectory() as temp, patch.object(fr, "MODES", TWO):
            for trial in range(2):
                output = Path(temp) / str(trial)
                output.mkdir()
                trial_manifest = copy.deepcopy(manifest)
                if trial:
                    for row in trial_manifest["test"]:
                        row["label"] = (row["label"] + 1) % 3

                def load(rows):
                    if any(int(row["file"]) >= 5 for row in rows):
                        self.assertTrue((output / "selection.json").is_file())
                    return images[[int(row["file"]) for row in rows]], torch.tensor([row["label"] for row in rows])

                def profile(r, model, *unused):
                    return (np.array([[4., 0.], [1., 0.]]), fr.cost_table(model), {"test": "same"},
                            {"shared_profile_build_seconds": 0.}, False)

                fake_runner = SimpleNamespace(
                    OUT=output, CHECKPOINT=Path(temp) / "checkpoint", BATCH_SIZE=2,
                    read_manifest=lambda: trial_manifest, select_device=lambda: torch.device("cpu"),
                    sync=lambda _: None, log=lambda _: None, load_images=load,
                    batch_forward=lambda model, x, policy, device: model(x, policy).detach(),
                    plot_frontier=lambda *a, **kw: None, pareto_indices=fr.pareto_indices,
                    device_runtime=lambda _: {"device": "cpu"})
                args = SimpleNamespace(budget_points=3, max_candidates=8, profile_cache=None,
                                       compute_fraction=None, router_mode="transfer")
                with patch.object(fr, "file_digest", return_value="checkpoint"), patch.object(fr, "load_model", side_effect=lambda *_: copy.deepcopy(template)), patch.object(fr, "prepare_profile", side_effect=profile):
                    fr.frontier(fake_runner, args)
                frozen.append(json.loads((output / "selection.json").read_text()))
                self.assertEqual(frozen[-1]["quantization_formats"], [list(m) for m in TWO])
            self.assertEqual(frozen[0], frozen[1])


if __name__ == "__main__":
    unittest.main()
