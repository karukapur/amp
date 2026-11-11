"""Mechanism check for mixed-precision residual transport sketches.

Synthetic, untrained residual MLP; this is not a ResNet/ImageNet benchmark.
Run: python3 transport_sketch_demo.py
Requires the already installed torch, numpy, scipy. No downloads or training.
"""

import itertools
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


MODES = ((4, 8), (4, 16), (8, 8), (8, 16))


def quantize(x, bits, bound):
    scale = bound.clamp_min(1e-12) / (2 ** (bits - 1) - 1)
    rounded = (x / scale).round().clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1) * scale
    # shortcut: transport uses a clipped STE, not derivatives of integer rounding.
    clipped = x.clamp(-bound, bound)
    return clipped + (rounded - clipped).detach()


def block(x, weight, mode, bound):
    if mode is None:
        return F.relu(x + F.linear(x, weight))
    wb, ab = MODES[mode]
    wq = quantize(weight, wb, weight.abs().amax(dim=1, keepdim=True))
    return quantize(F.relu(x + F.linear(x, wq)), ab, bound)


def forward(x, weights, head, modes, bounds):
    inputs, outputs = [], []
    for weight, mode, bound in zip(weights, modes, bounds):
        inputs.append(x)
        x = block(x, weight, mode, bound)
        outputs.append(x)
    return F.linear(x, head), inputs, outputs


def kl(teacher, logits):
    return (teacher * (teacher.clamp_min(1e-30).log() - logits.log_softmax(-1))).sum(-1).mean()


def score_table(linear, signatures, policies, baseline):
    indices = np.arange(len(linear))[None, :]
    chosen = signatures[indices, policies]
    first = linear[indices, policies].sum(axis=1)
    coupled = baseline + first + 0.5 * np.square(chosen.sum(axis=1)).sum(axis=1)
    additive = baseline + first + 0.5 * np.square(chosen).sum(axis=(1, 2))
    return coupled, additive


def covariance_self_check():
    rng = np.random.default_rng(19)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    noise = rng.normal(size=(100_000, len(p))) * np.sqrt(p)
    probes = noise - noise.sum(axis=1, keepdims=True) * p
    expected = np.diag(p) - np.outer(p, p)
    observed = probes.T @ probes / len(probes)
    assert np.max(np.abs(observed - expected)) < 0.004
    assert np.max(np.abs(probes.sum(axis=1))) < 1e-12

    # Equal scalar sensitivities can conceal complete cancellation.
    v = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert np.square(v.sum(axis=0)).sum() == 0
    assert np.square(v).sum() == 2


def diverse_shortlist(rankings, policies, limit=12):
    selected, per_weight_pattern = [], {}
    for candidates in itertools.zip_longest(*rankings):
        for candidate in candidates:
            if candidate is None or int(candidate) in selected:
                continue
            candidate = int(candidate)
            weight_pattern = tuple(MODES[i][0] for i in policies[candidate])
            if per_weight_pattern.get(weight_pattern, 0) >= 2:
                continue
            selected.append(candidate)
            per_weight_pattern[weight_pattern] = per_weight_pattern.get(weight_pattern, 0) + 1
            if len(selected) == limit:
                return selected
    return selected


def run(seed, anchor_mode, probes=8):
    torch.manual_seed(seed)
    n, width, classes, layers = 64, 24, 8, 5
    x = torch.randn(n, width, dtype=torch.float64)
    weights = [torch.randn(width, width, dtype=torch.float64) * (0.45 / math.sqrt(width)) for _ in range(layers)]
    head = torch.randn(classes, width, dtype=torch.float64) / math.sqrt(width)
    fp_modes = [None] * layers
    dummy_bounds = [torch.tensor(1.0)] * layers
    with torch.no_grad():
        teacher_logits, _, fp_outputs = forward(x, weights, head, fp_modes, dummy_bounds)
        teacher = teacher_logits.softmax(-1)
        bounds = [h.abs().max().detach() for h in fp_outputs]

    start = perf_counter()
    logits, inputs, outputs = forward(x.detach().requires_grad_(True), weights, head, [anchor_mode] * layers, bounds)
    probability = logits.detach().softmax(-1)
    baseline = float(kl(teacher, logits).detach())
    local_errors = []
    with torch.no_grad():
        for weight, bound, h_in, h_out in zip(weights, bounds, inputs, outputs):
            local_errors.append(torch.stack([block(h_in.detach(), weight, mode, bound) - h_out.detach() for mode in range(4)]))
    linear_gradient = torch.autograd.grad(kl(teacher, logits), outputs, retain_graph=True)
    linear = torch.stack([(error * grad[None]).sum(dim=(1, 2)) for error, grad in zip(local_errors, linear_gradient)]).numpy()
    signatures = torch.zeros(layers, 4, n, probes, dtype=torch.float64)
    for r in range(probes):
        scaled_noise = torch.randn_like(probability) * probability.sqrt()
        q = (scaled_noise - probability * scaled_noise.sum(-1, keepdim=True)).detach()
        gradients = torch.autograd.grad((q * logits).sum(), outputs, retain_graph=r + 1 < probes)
        for i, (error, gradient) in enumerate(zip(local_errors, gradients)):
            signatures[i, :, :, r] = (error * gradient[None]).sum(-1) / math.sqrt(n * probes)
    signatures = signatures.reshape(layers, 4, -1).numpy()
    profiling_seconds = perf_counter() - start
    assert np.allclose(signatures[:, anchor_mode], 0)
    assert np.allclose(linear[:, anchor_mode], 0)

    policies = np.array(list(itertools.product(range(4), repeat=layers)))
    wb = np.array([m[0] for m in MODES])[policies]
    ab = np.array([m[1] for m in MODES])[policies]
    # Equal-sized toy tensors: two independent storage budgets, not latency.
    feasible = (wb.sum(axis=1) <= 4 * layers + 4 * 2) & (ab.sum(axis=1) <= 8 * layers + 8 * 2)
    coupled, additive = score_table(linear, signatures, policies, baseline)
    available = np.flatnonzero(feasible)
    coupled_best = int(available[np.argmin(coupled[available])])
    additive_best = int(available[np.argmin(additive[available])])

    start = perf_counter()
    with torch.no_grad():
        exact = np.array([float(kl(teacher, forward(x, weights, head, modes, bounds)[0])) for modes in policies])
    exhaustive_seconds = perf_counter() - start
    actual_best = int(available[np.argmin(exact[available])])
    probe_curve = {}
    unflattened = signatures.reshape(layers, 4, n, probes)
    for count in (1, 2, 4, 8):
        prefix = unflattened[:, :, :, :count].reshape(layers, 4, -1) * math.sqrt(probes / count)
        prefix_score = score_table(linear, prefix, policies, baseline)[0]
        probe_curve[count] = round(float(spearmanr(prefix_score[available], exact[available]).statistic), 4)
    anchor_id = int(np.flatnonzero((policies == anchor_mode).all(axis=1))[0])
    assert abs(coupled[anchor_id] - exact[anchor_id]) < 1e-10

    # Equal-size shortlists get an independent selection set, then a third set
    # reports quality. Exhaustive profiling above is only a small-network oracle.
    audit_generator = torch.Generator().manual_seed(10_000 + seed)
    audit_x = torch.randn(256, width, generator=audit_generator, dtype=torch.float64)
    with torch.no_grad():
        audit_teacher = forward(audit_x, weights, head, fp_modes, dummy_bounds)[0].softmax(-1)
        audit_coupled = float(kl(audit_teacher, forward(audit_x, weights, head, policies[coupled_best], bounds)[0]))
        audit_additive = float(kl(audit_teacher, forward(audit_x, weights, head, policies[additive_best], bounds)[0]))
        coupled_rank = available[np.argsort(coupled[available])]
        additive_rank = available[np.argsort(additive[available])]
        rng = np.random.default_rng(20_000 + seed)
        shortlists = {
            "coupled_plus_additive": diverse_shortlist([additive_rank, coupled_rank], policies),
            "additive_only": diverse_shortlist([additive_rank], policies),
            "random": rng.choice(available, size=12, replace=False).tolist(),
        }
        audit_cache = {}
        for candidate in set(itertools.chain.from_iterable(shortlists.values())):
            audit_cache[candidate] = float(kl(audit_teacher, forward(audit_x, weights, head, policies[candidate], bounds)[0]))
        chosen = {name: min(candidates, key=audit_cache.get) for name, candidates in shortlists.items()}
        test_generator = torch.Generator().manual_seed(30_000 + seed)
        test_x = torch.randn(1_024, width, generator=test_generator, dtype=torch.float64)
        test_teacher = forward(test_x, weights, head, fp_modes, dummy_bounds)[0].softmax(-1)
        shortlist_test_kl = {name: float(kl(test_teacher, forward(test_x, weights, head, policies[candidate], bounds)[0])) for name, candidate in chosen.items()}
        assert all(len(candidates) == 12 for candidates in shortlists.values())

    # Verify the constant-time cached-sum swap identity used by the search.
    selected = policies[coupled_best].copy()
    total = signatures[np.arange(layers), selected].sum(axis=0)
    old, new = int(selected[0]), (int(selected[0]) + 1) % 4
    delta = signatures[0, new] - signatures[0, old]
    predicted_change = linear[0, new] - linear[0, old] + total @ delta + 0.5 * (delta @ delta)
    changed = selected.copy()
    changed[0] = new
    changed_score = score_table(linear, signatures, changed[None], baseline)[0][0]
    assert np.isclose(changed_score - coupled[coupled_best], predicted_change)

    # Exact finite-error transport for all local alternatives, via a tiny explicit
    # Jacobian. Used only as an independent audit of the randomized sketch.
    transported = torch.zeros(layers, 4, n, classes, dtype=torch.float64)
    for i in range(layers):
        for c in range(classes):
            state = outputs[i].detach().requires_grad_(True)
            suffix = state
            for j in range(i + 1, layers):
                suffix = block(suffix, weights[j], anchor_mode, bounds[j])
            class_logits = F.linear(suffix, head)
            gradient = torch.autograd.grad(class_logits[:, c].sum(), state)[0]
            transported[i, :, :, c] = (local_errors[i] * gradient[None]).sum(-1)
    selected_transport = transported.numpy()[np.arange(layers)[None, :], policies].sum(axis=1)
    p = probability.numpy()
    centered = selected_transport - (selected_transport * p[None]).sum(axis=-1, keepdims=True)
    quadratic = 0.5 * (np.square(centered) * p[None]).sum(axis=(1, 2)) / n
    exact_transport_score = baseline + linear[np.arange(layers)[None, :], policies].sum(axis=1) + quadratic
    return {
        "seed": seed,
        "anchor": MODES[anchor_mode],
        "probes_per_example": probes,
        "examples": n,
        "policies_evaluated": len(policies),
        "feasible_policies": len(available),
        "profile_seconds": round(profiling_seconds, 4),
        "exhaustive_evaluation_seconds": round(exhaustive_seconds, 4),
        "spearman_coupled_vs_actual_kl": round(float(spearmanr(coupled[available], exact[available]).statistic), 4),
        "spearman_additive_vs_actual_kl": round(float(spearmanr(additive[available], exact[available]).statistic), 4),
        "spearman_sketch_vs_exact_transport": round(float(spearmanr(coupled[available], exact_transport_score[available]).statistic), 4),
        "probe_count_vs_policy_rank_correlation": probe_curve,
        "actual_kl_coupled_choice": float(exact[coupled_best]),
        "actual_kl_additive_choice": float(exact[additive_best]),
        "actual_kl_exhaustive_best": float(exact[actual_best]),
        "disjoint_examples": len(audit_x),
        "disjoint_kl_coupled_choice": audit_coupled,
        "disjoint_kl_additive_choice": audit_additive,
        "shortlist_policy_evaluations_per_method": 12,
        "final_test_examples": len(test_x),
        "shortlist_selected_final_test_kl": shortlist_test_kl,
        "coupled_policy_w_a": [MODES[i] for i in policies[coupled_best]],
        "additive_policy_w_a": [MODES[i] for i in policies[additive_best]],
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    covariance_self_check()
    results = [run(seed, anchor) for seed in range(3) for anchor in (3, 0)]
    output = {
        "scope": "Synthetic untrained five-block residual MLP. Tests transport/sketch mechanism; no ImageNet accuracy or deployment speed claims.",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "checks": ["Fisher probe covariance", "softmax constant-shift nullspace", "signed cancellation", "zero anchor error", "exact anchor score", "cached swap identity"],
        "runs": results,
    }
    path = Path(__file__).with_name("transport_sketch_results.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
