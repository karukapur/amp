"""Independent numerical checks and search invariants for layerwise PTQ."""
import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from amp import fisher_router as fr, runner


class ResidualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(5, 6)
        self.b = nn.Linear(6, 6)
        self.shortcut = nn.Linear(5, 6)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        output = self.b(self.relu(self.a(x)))
        identity = self.shortcut(x)
        output += identity  # Match torchvision's in-place residual addition.
        return self.head(self.relu(output))


class FisherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        self.model = fr.LayerModel(ResidualClassifier()).eval()
        self.x, self.y = torch.randn(7, 5), torch.tensor([0, 2, 1, 2, 0, 1, 2])
        fr.calibrate(self.model, [(self.x, self.y)], torch.device("cpu"))

    def test_fisher_matches_direct_per_example_autograd_and_microbatches(self):
        saved = {k: v.clone() for k, v in self.model.state_dict().items()}
        expected = np.zeros((4, 5))
        for x, label in zip(self.x, self.y):
            inputs, outputs, handles = [], [], []
            def retain_unmodified_output(module, args, output):
                inputs.append(args[0].detach())
                outputs.append(output)
                # The direct-autograd oracle differentiates the pre-addition tensor.
                return output.clone()
            for op in self.model.operators.values():
                handles.append(op.register_forward_hook(retain_unmodified_output))
            logits = self.model(x[None].requires_grad_())
            for h in handles:
                h.remove()
            gradients = torch.autograd.grad(F.cross_entropy(logits, label[None]), outputs)
            with torch.no_grad():
                for i, (op, inp, out, grad) in enumerate(zip(self.model.operators.values(), inputs, outputs, gradients)):
                    for mode, (w, a) in enumerate(fr.MODES[:4]):
                        bound = op.input_quantizer.bound
                        scale = bound / (2 ** (a - 1) - 1)
                        quant_x = (inp.clamp(-bound, bound) / scale).round() * scale
                        quant_y = F.linear(quant_x, getattr(op.affine, f"w{w}"), op.affine.op.bias)
                        expected[i, mode] += float(((quant_y - out) * grad).clamp(-1e10, 1e10).square().sum()) / len(self.x)
        # Execution order is a, b, shortcut, head, matching registered operator order.
        for batch_size in (1, 3, 7):
            batches = [(self.x[i:i + batch_size], self.y[i:i + batch_size]) for i in range(0, len(self.x), batch_size)]
            result = fr.fisher_scores(self.model, batches, torch.device("cpu"))
            np.testing.assert_allclose(result, expected, atol=2e-10, rtol=3e-4)
            self.assertTrue(np.array_equal(result[:, fr.FP32], np.zeros(4)))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, saved[name]), name)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))
        self.assertTrue(all(not op._forward_hooks for op in self.model.operators.values()))

    def test_profile_hooks_cleanup_on_failure(self):
        with self.assertRaises((IndexError, RuntimeError)):
            fr.fisher_scores(self.model, [(self.x, torch.full_like(self.y, 99))], torch.device("cpu"))
        self.assertTrue(all(not op._forward_hooks for op in self.model.operators.values()))
        self.assertEqual(self.model(self.x).shape, (7, 3))

    def test_costs_include_fp32_metadata_and_activation_compute(self):
        costs = fr.cost_table(self.model)
        for i, op in enumerate(self.model.operators.values()):
            params = op.affine.op.weight.numel()
            biases = op.affine.op.bias.numel()
            self.assertEqual(costs[i, 4, 0], 4 * (params + biases))
            self.assertEqual(costs[i, 0, 0], (params * 4 + 7) // 8 + 4 * biases + 4 * op.affine.op.weight.shape[0] + 4)
            self.assertEqual(costs[i, 0, 0], costs[i, 1, 0])
            self.assertEqual(costs[i, 1, 1], 2 * costs[i, 0, 1])
            self.assertEqual(costs[i, 4, 1], 16 * costs[i, 2, 1])

    def test_policy_reload_logits_csv_and_validation(self):
        policy = (0, 1, 2, 4)
        before = self.model(self.x, policy).detach()
        saved = fr.policy_document(self.model, policy, "checkpoint")
        self.model.restore_bounds({n: 1.0 for n in self.model.operators})
        restored = fr.restore_policy(self.model, json.loads(json.dumps(saved)), "checkpoint")
        self.assertTrue(torch.equal(before, self.model(self.x, restored)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.csv"
            fr.export_policy(self.model, policy, path)
            import csv
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[-1]["notes"], "FP32 bypass")
            self.assertEqual(rows[3]["activation_bits"], "16")
        for bad in (dict(saved, checkpoint_sha256="wrong"), dict(saved, graph="wrong"), dict(saved, schema_version=999)):
            with self.assertRaises(ValueError):
                fr.restore_policy(self.model, bad, "checkpoint")
        bad = copy.deepcopy(saved)
        bad["assignments"]["a"]["activation_bits"] = 7
        with self.assertRaises(ValueError):
            fr.restore_policy(self.model, bad, "checkpoint")
        with self.assertRaises(ValueError):
            self.model.restore_bounds({n: float("nan") for n in self.model.operators})
        for invalid in ((True, 1, 2, 3), (0, 1, 2), (0, 1, 2, 5)):
            with self.assertRaises(ValueError):
                fr.validate_policy(invalid, 4)

    def test_profile_cache_rejects_changed_inputs_and_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            metadata = {"fingerprint": "original"}
            scores = np.zeros((4, 5))
            costs = fr.cost_table(self.model)
            payload = {"metadata": metadata, "activation_bounds": self.model.bounds(), "scores": scores.tolist(),
                       "costs": costs.tolist(), "operator_macs": {n: op.affine.macs for n, op in self.model.operators.items()}}
            fake_runner = SimpleNamespace(log=lambda _: None)
            fr.write_json(path, payload)
            with patch.object(fr, "profile_metadata", return_value=metadata), patch.object(fr, "calibrate") as calibration, patch.object(fr, "fisher_scores") as profiling:
                restored = fr.prepare_profile(fake_runner, self.model, {}, "cpu", "hash", path)
                self.assertTrue(restored[-1])
                calibration.assert_not_called()
                profiling.assert_not_called()
            with patch.object(fr, "profile_metadata", return_value={"fingerprint": "changed"}), self.assertRaisesRegex(ValueError, "fingerprint"):
                fr.prepare_profile(fake_runner, self.model, {}, "cpu", "hash", path)
            payload["costs"][0][0][0] += 1
            fr.write_json(path, payload)
            with patch.object(fr, "profile_metadata", return_value=metadata), self.assertRaisesRegex(ValueError, "cost table"):
                fr.prepare_profile(fake_runner, self.model, {}, "cpu", "hash", path)


class RouterTests(unittest.TestCase):
    def tables(self):
        scores = np.array([[0., 10, 1, 10, 10], [0, 10, 0, 10, 10], [0, 10, 20, 10, 10]])
        costs = np.tile(np.array([[1, 1], [1, 2], [2, 2], [2, 4], [4, 16]], dtype=float), (3, 1, 1))
        return scores, costs

    def test_storage_and_bitops_shortlists_keep_different_tradeoffs(self):
        policies = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0)]
        values = [([10, 100], 2, .1, True), ([10, 50], 3, .1, True),
                  ([20, 200], 1, .1, True), ([20, 50], 3, .2, True),
                  ([1, 1], 0, .01, False), ([30, 300], 4, .1, True)]
        records = [dict(policy=list(p), cost=cost, errors=errors, mean_kl=kl, complete=complete)
                   for p, (cost, errors, kl, complete) in zip(policies, values)]
        storage = fr.search_frontier_policies(records, 0)
        bitops = fr.search_frontier_policies(records, 1)
        self.assertEqual(storage, {policies[0], policies[2]})
        self.assertEqual(bitops, {policies[0], policies[1], policies[2]})
        self.assertIn(policies[1], storage | bitops)  # A storage-only shortlist would lose it.
        self.assertNotIn(policies[4], storage | bitops)  # Incomplete evaluation cannot qualify.
        self.assertEqual(fr.search_frontier_policies([], 1), set())

    def test_ilp_matches_exhaustive_under_both_constraints(self):
        rng = np.random.default_rng(28)
        scores, costs = self.tables()
        scores = rng.random(scores.shape)
        for budget in ([3, 3], [4, 5], [7, 8]):
            policies = [p for p in itertools.product(range(5), repeat=3) if fr.feasible(fr.policy_cost(costs, p), budget)]
            expected = min(float(scores[np.arange(3), p].sum()) for p in policies)
            result, status = fr.solve_seed(scores, costs, budget)
            self.assertTrue(status["optimal"])
            self.assertTrue(fr.feasible(fr.policy_cost(costs, result), budget))
            self.assertAlmostEqual(float(scores[np.arange(3), result].sum()), expected)
        policy, status = fr.solve_seed(scores, costs, [1, 1])
        self.assertIsNone(policy)
        self.assertFalse(status["optimal"])

    def test_storage_only_ilp_matches_exhaustive_and_allows_higher_activation_precision(self):
        _, costs = self.tables()
        scores = np.tile([8., 0, 4, 1, 0], (3, 1))
        for budget in ([3], [4], [7]):
            policies = [p for p in itertools.product(range(5), repeat=3)
                        if fr.policy_cost(costs, p)[0] <= budget[0]]
            expected = min(float(scores[np.arange(3), p].sum()) for p in policies)
            policy, report = fr.solve_seed(scores, costs, budget)
            self.assertTrue(report["optimal"])
            self.assertLessEqual(fr.policy_cost(costs, policy)[0], budget[0])
            self.assertAlmostEqual(float(scores[np.arange(3), policy].sum()), expected)
            self.assertEqual(report["budget"], budget)
        uncapped, _ = fr.solve_seed(scores, costs, [3])
        old_cap = .75 * fr.policy_cost(costs, (2, 2, 2))[1]
        self.assertEqual(uncapped, (1, 1, 1))  # W4A16 wins at the same storage as W4A8.
        self.assertGreater(fr.policy_cost(costs, uncapped)[1], old_cap)
        capped, _ = fr.solve_seed(scores, costs, [3, old_cap])
        self.assertLessEqual(fr.policy_cost(costs, capped)[1], old_cap)
        self.assertGreater(float(scores[np.arange(3), capped].sum()), 0)
        for invalid in ([], [3, 4, 5], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                fr.solve_seed(scores, costs, invalid)

    def test_storage_only_neighbors_allow_compute_increases(self):
        scores, costs = self.tables()
        proposals = list(fr.neighbors((0, 0, 0), scores, costs, [3], False))
        self.assertIn((1, 0, 0), proposals)
        self.assertIn((1, 1, 0), proposals)
        for policy in proposals:
            self.assertLessEqual(fr.policy_cost(costs, policy)[0], 3)
        self.assertFalse(list(fr.neighbors((0, 0, 0), scores, costs, [3, 3], False)))

    def test_storage_only_transfers_and_incumbents_ignore_unconstrained_bitops(self):
        scores, costs = self.tables()
        costs[:, :, 1] *= 100
        seeds = [(2, 0, 0), (2, 2, 0)]
        errors = {seeds[0]: 6, seeds[1]: 4, (0, 0, 0): 5, (0, 2, 0): 3}
        def evaluate(policy, cutoff):
            return {"errors": errors.get(policy, 9), "mean_kl": .1, "images": 10, "complete": True}
        result = fr.run_search(scores, costs, [[4], [5]], seeds, evaluate, max_candidates=4)
        self.assertEqual([r["kind"] for r in result["records"]], ["seed", "seed", "local", "transfer"])
        self.assertEqual(result["incumbents"], [[0, 2, 0], [0, 2, 0]])
        self.assertTrue(all(r["cost"][1] >= 300 for r in result["records"]))
        self.assertTrue(all(fr.feasible(r["cost"], [5]) for r in result["records"]))
        self.assertFalse(fr.feasible(result["records"][-1]["cost"], [5, 10]))

    def test_successful_edit_routes_to_another_budget(self):
        scores, costs = self.tables()
        original = scores.copy()
        seeds = [(2, 0, 0), (2, 2, 0)]
        budgets = [[4, 10], [5, 10]]
        errors = {seeds[0]: 6, seeds[1]: 4, (0, 0, 0): 5, (0, 2, 0): 3}
        calls = []
        def evaluate(p, cutoff):
            calls.append((p, cutoff))
            return {"errors": errors.get(p, 9), "mean_kl": .1, "images": 10, "complete": True}
        result = fr.run_search(scores, costs, budgets, seeds, evaluate, max_candidates=4)
        self.assertEqual([r["kind"] for r in result["records"]], ["seed", "seed", "local", "transfer"])
        self.assertEqual(result["records"][-1]["policy"], [0, 2, 0])
        self.assertEqual(result["records"][-1]["transfer_id"], 0)
        self.assertEqual(result["incumbents"], [[0, 2, 0], [0, 2, 0]])
        self.assertEqual(calls[2][1], 6)  # Must beat every eligible incumbent to reject safely.
        self.assertEqual(len({p for p, _ in calls}), 4)
        np.testing.assert_array_equal(scores, original)
        for record in result["records"]:
            self.assertTrue(any(fr.feasible(record["cost"], b) for b in budgets))

    def test_dedup_budget_ties_and_local_ablation(self):
        scores, costs = self.tables()
        calls = []
        def evaluate(p, cutoff):
            calls.append(p)
            return {"errors": 2, "mean_kl": .1, "images": 8, "complete": True}
        result = fr.run_search(scores, costs, [[4, 10], [5, 10]], [(0, 0, 0)] * 2, evaluate, mode="ilp", max_candidates=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["policy_image_evaluations"], 8)
        result = fr.run_search(scores, costs, [[4, 10], [5, 10]], [(0, 0, 0)] * 2, evaluate, mode="local", max_candidates=5)
        self.assertEqual(result["candidate_count"], 5)
        self.assertFalse(any(r["kind"] == "transfer" for r in result["records"]))
        self.assertLess(fr.metric({"errors": 2, "mean_kl": .1}), fr.metric({"errors": 2, "mean_kl": .2}))
        self.assertLess(fr.metric({"errors": 1, "mean_kl": 10}), fr.metric({"errors": 2, "mean_kl": .1}))

    def test_failed_transfer_does_not_ban_an_edit_in_another_context(self):
        scores, costs = self.tables()
        seeds = [(2, 0, 0), (2, 2, 0), (2, 2, 2)]
        errors = {seeds[0]: 6, seeds[1]: 4, seeds[2]: 2, (0, 0, 0): 5, (0, 2, 0): 7, (0, 2, 2): 1}
        def evaluate(policy, cutoff):
            return {"errors": errors.get(policy, 9), "mean_kl": .1, "images": 10, "complete": True}
        result = fr.run_search(scores, costs, [[4, 10], [5, 10], [6, 10]], seeds, evaluate, max_candidates=6)
        routed = [r for r in result["records"] if r["kind"] == "transfer"]
        self.assertEqual([r["policy"] for r in routed], [[0, 2, 0], [0, 2, 2]])
        self.assertEqual([r["errors"] for r in routed], [7, 1])
        self.assertEqual([r["transfer_id"] for r in routed], [0, 0])

    def test_budget_must_cover_all_unique_seeds(self):
        scores, costs = self.tables()
        with self.assertRaisesRegex(ValueError, "allowance"):
            fr.run_search(scores, costs, [[4, 10], [5, 10]], [(2, 0, 0), (2, 2, 0)], lambda *_: None, max_candidates=1)

    def test_neighbors_and_transfer_require_compatible_choices(self):
        scores, costs = self.tables()
        parent, budget = (2, 0, 0), [4, 5]
        proposals = list(fr.neighbors(parent, scores, costs, budget, True))
        self.assertEqual(len(proposals), len(set(proposals)))
        for p in proposals:
            self.assertIn(len(fr.edit_between(parent, p)), (1, 2))
            self.assertTrue(fr.feasible(fr.policy_cost(costs, p), budget))
        self.assertEqual(fr.apply_edit((2, 2, 0), [(0, 2, 0)]), (0, 2, 0))
        self.assertIsNone(fr.apply_edit((1, 2, 0), [(0, 2, 0)]))

    def test_early_rejection_requires_strictly_more_errors_and_completes_ties(self):
        class IndexedModel:
            def __call__(self, x, policy=None):
                logits = torch.zeros(len(x), 2)
                predictions = x[:, 0].long()
                logits[torch.arange(len(x)), predictions] = 3
                return logits
        x = torch.tensor([[1.], [1.], [0.], [1.], [0.], [0.]])
        labels = torch.zeros(6, dtype=torch.long)
        teacher = torch.tensor([[.9, .1]]).repeat(6, 1)
        evaluate = fr.search_evaluator(IndexedModel(), x, labels, teacher, "cpu", 2, lambda _: None)
        rejected = evaluate((0,), 2)
        self.assertEqual((rejected["images"], rejected["errors"], rejected["complete"]), (4, 3, False))
        tied = evaluate((0,), 3)
        self.assertEqual((tied["images"], tied["errors"], tied["complete"]), (6, 3, True))

    def test_layer_dispatch_preserves_legacy_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps({"granularity": "operator"}))
            with patch.object(fr, "evaluate_saved") as evaluate:
                runner.evaluate_saved(path, export_only=True)
                evaluate.assert_called_once_with(runner, {"granularity": "operator"}, True)


if __name__ == "__main__":
    unittest.main()
