"""Layerwise PTQ with Fisher scoring and measured cross-budget edit transfer.

The scoring functional is adapted from NVIDIA Model Optimizer (Apache-2.0).
See third_party/modelopt/NOTICE.md. Other search code is a local experiment.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import Bounds, LinearConstraint, milp
from torchvision.models import resnet50

from .runner import Activation, Affine, LOGGER, pareto_indices


MODES = (
    (4, 8),    # W4A8
    (4, 16),   # W4A16
    (8, 8),    # W8A8
    (8, 16),   # W8A16
)
VERSION = 1
QUANTIZATION = "symmetric_maxabs_per_channel_weights_per_input_activations_ties_even_v1"
UPSTREAM = "814e6b49ea6752498b5443d49bc8d39a3ca6ad65"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_policy(policy, count):
    if not isinstance(policy, (list, tuple, np.ndarray)) or len(policy) != count:
        raise ValueError(f"Layer policy requires {count} mode IDs.")
    if any(isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or not 0 <= m < len(MODES) for m in policy):
        raise ValueError(f"Layer mode IDs must be integers from 0 through {len(MODES) - 1}; FP32 is a baseline only.")
    return tuple(int(m) for m in policy)


class QuantOperator(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.input_quantizer = Activation()
        self.affine = Affine(op)
        self.mode = None

    def compute(self, x, mode):
        w, a = (None, None) if mode is None else MODES[mode]
        return self.affine(self.input_quantizer(x, a), w)

    def forward(self, x):
        return self.compute(x, self.mode)


class LayerModel(nn.Module):
    """Explicit consumer-input quantizers; identity residual edges stay floating."""
    def __init__(self, network):
        super().__init__()
        self.network = network.eval().requires_grad_(False)
        for module in self.network.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False  # Backward hooks must see unmodified operator outputs.
        for name, module in list(self.network.named_modules()):
            if name and isinstance(module, (nn.Conv2d, nn.Linear)):
                parent, _, child = name.rpartition(".")
                setattr(self.network.get_submodule(parent), child, QuantOperator(module))
        self.operators = dict((n, m) for n, m in self.network.named_modules() if isinstance(m, QuantOperator))
        if not self.operators:
            raise ValueError("No convolution or linear operators found.")

    def forward(self, x, policy=None):
        policy = (None,) * len(self.operators) if policy is None else validate_policy(policy, len(self.operators))
        for operator, mode in zip(self.operators.values(), policy):
            operator.mode = mode
        return self.network(x)

    def bounds(self):
        return {n: float(m.input_quantizer.bound.detach().cpu()) for n, m in self.operators.items()}

    def restore_bounds(self, bounds):
        if set(bounds) != set(self.operators):
            raise ValueError("Saved activation quantizers do not match the layer graph.")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in bounds.values()):
            raise ValueError("Activation bounds must be positive finite numbers.")
        for name, value in bounds.items():
            self.operators[name].input_quantizer.bound.fill_(value)


def load_model(checkpoint, device):
    network = resnet50(weights=None).eval()
    network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    network.conv1 = torch.nn.utils.fuse_conv_bn_eval(network.conv1, network.bn1)
    network.bn1 = nn.Identity()
    for stage in (network.layer1, network.layer2, network.layer3, network.layer4):
        for block in stage:
            for k in (1, 2, 3):
                setattr(block, f"conv{k}", torch.nn.utils.fuse_conv_bn_eval(getattr(block, f"conv{k}"), getattr(block, f"bn{k}")))
                setattr(block, f"bn{k}", nn.Identity())
            if block.downsample is not None:
                block.downsample[0] = torch.nn.utils.fuse_conv_bn_eval(block.downsample[0], block.downsample[1])
                block.downsample[1] = nn.Identity()
    return LayerModel(network).eval().to(device)


def calibrate(model, batches, device):
    for op in model.operators.values():
        op.input_quantizer.bound.fill_(1e-5)
        op.input_quantizer.observe = True
    try:
        with torch.no_grad():
            for x, _ in batches:
                model(x.to(device))
    finally:
        for op in model.operators.values():
            op.input_quantizer.observe = False


def fisher_scores(model, batches, device):
    """One summed-CE backward per batch; replay candidates at each output hook."""
    scores = torch.zeros(len(model.operators), len(MODES), device=device)
    handles, output_handles, image_count = [], [], 0

    def capture(index):
        def hook(module, inputs, output):
            # Torchvision bottlenecks later add the shortcut to conv3's output in place.
            x, reference = inputs[0].detach(), output.detach().clone()

            def score(gradient):
                with torch.no_grad():
                    if not torch.isfinite(gradient).all():
                        raise ValueError("Non-finite output gradient during Fisher profiling.")
                    for mode in range(len(MODES)):
                        difference = module.compute(x, mode) - reference
                        if not torch.isfinite(difference).all():
                            raise ValueError("Non-finite candidate output during Fisher profiling.")
                        # NVIDIA CORPORATION & AFFILIATES, Copyright (c) 2024.
                        # Adapted _get_auto_quantize_score; Apache-2.0 (see NOTICE).
                        product = gradient.float() * difference.float()
                        scores[index, mode].add_(product.clamp(-1e10, 1e10).square().sum())
            output_handles.append(output.register_hook(score))
        return hook

    try:
        for index, op in enumerate(model.operators.values()):
            handles.append(op.register_forward_hook(capture(index)))
        with torch.enable_grad():
            for x, labels in batches:
                try:
                    logits = model(x.to(device).detach().requires_grad_(True))
                    loss = F.cross_entropy(logits, labels.to(device), reduction="sum")
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite profiling loss.")
                    loss.backward()
                finally:
                    # Detached output snapshots must not survive into the next batch.
                    for handle in output_handles:
                        handle.remove()
                    output_handles.clear()
                image_count += len(x)
                del logits, loss
    finally:
        for handle in handles:
            handle.remove()
    if not image_count or not torch.isfinite(scores).all():
        raise ValueError("Fisher profiling needs nonempty data and finite scores.")
    return scores.cpu().double().numpy() / image_count


def cost_table(model):
    """Estimated packed bytes and MAC BitOps for searchable quantized formats."""
    costs = np.zeros((len(model.operators), len(MODES), 2), dtype=np.float64)
    for i, op in enumerate(model.operators.values()):
        affine = op.affine
        if affine.macs <= 0:
            raise ValueError("Run a reference forward before computing costs.")
        bias = 4 * (affine.op.bias.numel() if affine.op.bias is not None else 0)
        for mode, (w, a) in enumerate(MODES):
            metadata = 4 * affine.op.weight.shape[0] + 4
            costs[i, mode] = (math.ceil(affine.count * w / 8) + bias + metadata, affine.macs * w * a)
    return costs


def fp32_reference_cost(model):
    """Cost of the separate floating-point baseline, outside the policy space."""
    total = np.zeros(2, dtype=np.float64)
    for op in model.operators.values():
        affine = op.affine
        if affine.macs <= 0:
            raise ValueError("Run a reference forward before computing costs.")
        bias_count = affine.op.bias.numel() if affine.op.bias is not None else 0
        total += (4 * (affine.count + bias_count), affine.macs * 32 * 32)
    return total


def policy_cost(costs, policy):
    return costs[np.arange(len(policy)), policy].sum(axis=0)


def feasible(cost, budget):
    # Budgets contain storage alone, or storage followed by an explicit BitOps cap.
    return bool(np.all(np.asarray(cost)[:len(budget)] <= np.asarray(budget) + 1e-9))


def solve_seed(scores, costs, budget):
    layers, formats = scores.shape
    if formats != len(MODES) or costs.shape != (layers, formats, 2):
        raise ValueError("Sensitivity/cost tables must contain only the configured quantized formats; FP32 is a baseline only.")
    budget = np.asarray(budget, dtype=float)
    if (budget.ndim != 1 or budget.size not in (1, 2) or not np.isfinite(budget).all()
            or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(budget <= 0)):
        raise ValueError("Invalid scores or resource budget.")
    resource_costs = (costs[:, :, :len(budget)] / budget).reshape(layers * formats, len(budget)).T
    constraints = np.vstack((np.repeat(np.eye(layers), formats, axis=1), resource_costs))
    low = np.r_[np.ones(layers), np.full(len(budget), -np.inf)]
    high = np.ones(layers + len(budget))
    start = perf_counter()
    result = milp(scores.ravel() / max(float(scores.max()), 1e-30), integrality=np.ones(scores.size),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(constraints, low, high),
                  options={"time_limit": 10.0, "mip_rel_gap": 0.0})
    report = {"status": int(result.status), "message": result.message, "seconds": perf_counter() - start,
              "optimal": bool(result.status == 0), "budget": list(map(float, budget))}
    if result.x is None:
        return None, report
    choices = result.x.reshape(layers, formats)
    policy = tuple(map(int, choices.argmax(axis=1)))
    expected = np.eye(formats)[list(policy)]
    if not np.allclose(choices, expected, atol=1e-5, rtol=0) or not feasible(policy_cost(costs, policy), budget):
        report["invalid_incumbent"] = True
        return None, report
    report.update(policy=list(policy), score=float(scores[np.arange(layers), policy].sum()),
                  actual_cost=policy_cost(costs, policy).tolist())
    return policy, report


def metric(record):
    return record["errors"], record["mean_kl"]


def search_frontier_policies(records, cost_index):
    """Freeze the measured search frontier for storage (0) or BitOps (1)."""
    by_cost = {}
    for record in records:
        if not record["complete"]:
            continue
        cost = record["cost"][cost_index]
        if cost not in by_cost or (metric(record), record["policy"]) < (metric(by_cost[cost]), by_cost[cost]["policy"]):
            by_cost[cost] = record
    winners = list(by_cost.values())
    front = pareto_indices([r["cost"][cost_index] for r in winners], [r["errors"] for r in winners])
    return {tuple(winners[i]["policy"]) for i in front}


def edit_between(before, after):
    return tuple((i, a, b) for i, (a, b) in enumerate(zip(before, after)) if a != b)


def apply_edit(policy, edit):
    if any(policy[i] != old for i, old, _ in edit):
        return None
    candidate = list(policy)
    for i, _, new in edit:
        candidate[i] = new
    return tuple(candidate)


def neighbors(policy, scores, costs, budget, prefer_weight):
    """Rank O(L²K²) cheap edits, never enumerate whole-network assignments."""
    moves = [(i, m) for i, current in enumerate(policy) for m in range(len(MODES)) if m != current]
    layer, mode = np.array(moves).T
    old = np.array(policy)[layer]
    ds = scores[layer, mode] - scores[layer, old]
    dc = costs[layer, mode] - costs[layer, old]
    changes_weight = np.array([MODES[m][0] != MODES[o][0] for m, o in zip(mode, old)])
    left, right = np.triu_indices(len(moves), 1)
    different = layer[left] != layer[right]
    left, right = left[different], right[different]
    single_count = len(moves)
    delta_score = np.r_[ds, ds[left] + ds[right]]
    delta_cost = np.vstack((dc, dc[left] + dc[right]))
    weight = np.r_[changes_weight, changes_weight[left] | changes_weight[right]]
    valid = np.flatnonzero(np.all(policy_cost(costs, policy)[:len(budget)] + delta_cost[:, :len(budget)] <= np.asarray(budget) + 1e-9, axis=1))
    order = valid[np.lexsort((valid, delta_score[valid], weight[valid] != prefer_weight))]
    for k in order:
        candidate = list(policy)
        selected = [k] if k < single_count else [left[k - single_count], right[k - single_count]]
        for j in selected:
            candidate[layer[j]] = int(mode[j])
        yield tuple(candidate)


def run_search(scores, costs, budgets, seeds, evaluator, mode="transfer", max_candidates=64, on_record=None):
    """No dataset access: evaluator only receives a policy and safe error cutoff."""
    if mode not in ("ilp", "local", "transfer"):
        raise ValueError("Unknown router mode.")
    if scores.shape[1] != len(MODES) or costs.shape != (*scores.shape, 2):
        raise ValueError("Search tables must contain only the configured quantized formats.")
    seeds = [validate_policy(policy, len(scores)) for policy in seeds]
    if max_candidates < len(set(seeds)):
        raise ValueError("Candidate allowance must cover all unique ILP seeds.")
    archive, transfers = {}, []
    incumbents = [None] * len(budgets)
    local_turns = [0] * len(budgets)
    cursor, routing_seconds, search_seconds = 0, 0.0, 0.0
    neighbor_cache = {}

    def eligible(policy):
        value = policy_cost(costs, policy)
        return [i for i, budget in enumerate(budgets) if feasible(value, budget)]

    def evaluate(policy, kind, target=None, parent=None, transfer_id=None):
        nonlocal search_seconds
        slots = eligible(policy)
        if not slots:
            raise ValueError("Attempted to evaluate an infeasible policy.")
        cutoff = None if kind == "seed" or any(incumbents[i] is None for i in slots) else max(archive[incumbents[i]]["errors"] for i in slots)
        LOGGER.info("[search] Candidate %d/%d started: kind=%s, target=%s, transfer=%s, error_cutoff=%s, eligible_budgets=%s.",
                    len(archive) + 1, max_candidates, kind, f"B{target + 1}" if target is not None else "seed",
                    f"E{transfer_id + 1}" if transfer_id is not None else "none", cutoff,
                    ",".join(f"B{i + 1}" for i in slots))
        start = perf_counter()
        result = evaluator(policy, cutoff)
        elapsed = perf_counter() - start
        search_seconds += elapsed
        result = dict(result, id=len(archive), policy=list(policy), kind=kind, target_budget=target,
                      parent=list(parent) if parent is not None else None, transfer_id=transfer_id,
                      cutoff=cutoff, seconds=elapsed, cost=policy_cost(costs, policy).tolist())
        if result["complete"] and (not math.isfinite(result["mean_kl"]) or result["errors"] < 0):
            raise ValueError("Invalid candidate measurement.")
        archive[policy] = result
        updated = []
        if result["complete"]:
            for slot in slots:
                previous = incumbents[slot]
                if previous is None or (metric(result), policy) < (metric(archive[previous]), previous):
                    incumbents[slot] = policy
                    updated.append(f"B{slot + 1}")
            if parent is not None and metric(result) < metric(archive[parent]):
                edit = edit_between(parent, policy)
                transfers.append({"id": len(transfers), "source_budget": target, "source_policy": list(parent),
                                  "result_policy": list(policy), "edit": [list(e) for e in edit],
                                  "error_gain": archive[parent]["errors"] - result["errors"],
                                  "kl_gain": archive[parent]["mean_kl"] - result["mean_kl"]})
                edit_summary = ", ".join(f"L{i + 1}: W{MODES[old][0]}A{MODES[old][1]} -> W{MODES[new][0]}A{MODES[new][1]}"
                                         for i, old, new in edit)
                LOGGER.info("[router] Learned edit E%d: %s; source=B%d, error_gain=%d, KL_gain=%.6g.",
                            transfers[-1]["id"] + 1, edit_summary, target + 1,
                            transfers[-1]["error_gain"], transfers[-1]["kl_gain"])
        LOGGER.info("[search] Candidate %d/%d %s in %.2fs: mistakes=%d/%d, mean_KL=%.6g, updated=%s, storage=%.3f MB, BitOps=%.3f G.",
                    result["id"] + 1, max_candidates, "completed" if result["complete"] else "early rejected",
                    elapsed, result["errors"], result["images"], result["mean_kl"], ",".join(updated) or "none",
                    result["cost"][0] / 1e6, result["cost"][1] / 1e9)
        if on_record:
            on_record(result)

    for seed in seeds:
        if seed not in archive:
            evaluate(seed, "seed")
    if any(p is None for p in incumbents):
        raise ValueError("Every budget needs a complete feasible seed evaluation.")

    def local_proposal():
        nonlocal cursor
        for _ in budgets:
            slot = cursor % len(budgets)
            cursor += 1
            parent = incumbents[slot]
            preference = local_turns[slot] % 2 == 0
            cache_key = (slot, parent, preference)
            if cache_key not in neighbor_cache:
                # Discard stale parent iterators, retaining only current incumbents.
                for key in list(neighbor_cache):
                    if key[0] == slot and key[1] != parent:
                        del neighbor_cache[key]
                neighbor_cache[cache_key] = neighbors(parent, scores, costs, budgets[slot], preference)
            for candidate in neighbor_cache[cache_key]:
                if candidate not in archive:
                    local_turns[slot] += 1
                    return candidate, "local", slot, parent, None
        return None

    def transfer_proposal():
        options = []
        for transfer in transfers:
            edit = transfer["edit"]
            changed = {i for i, _, _ in edit}
            for slot, parent in enumerate(incumbents):
                if slot == transfer["source_budget"]:
                    continue
                candidate = apply_edit(parent, edit)
                if candidate is None or candidate in archive or not feasible(policy_cost(costs, candidate), budgets[slot]):
                    continue
                distance = sum(a != b for i, (a, b) in enumerate(zip(parent, transfer["source_policy"])) if i not in changed)
                rank = (distance, -transfer["error_gain"], -transfer["kl_gain"], candidate, slot, transfer["id"])
                options.append((rank, (candidate, "transfer", slot, parent, transfer["id"])))
        return min(options, key=lambda item: item[0])[1] if options else None

    step = 0
    stop_reason = "ILP seeds evaluated" if mode == "ilp" else "candidate allowance reached"
    while mode != "ilp" and len(archive) < max_candidates:
        start = perf_counter()
        proposal = transfer_proposal() if mode == "transfer" and (step + 1) % 4 else None
        if proposal is None:
            proposal = local_proposal()
        routing_seconds += perf_counter() - start
        if proposal is None:
            stop_reason = "no feasible untested proposal remains"
            LOGGER.info("[router] No feasible untested proposal remains.")
            break
        evaluate(*proposal)
        step += 1
    LOGGER.info("[search] Finished mode=%s (%s): %d unique candidates, %d learned edits, %d policy-image evaluations; routing=%.2fs, evaluation=%.2fs.",
                mode, stop_reason, len(archive), len(transfers), sum(r["images"] for r in archive.values()), routing_seconds, search_seconds)
    return {"records": list(archive.values()), "transfers": transfers,
            "incumbents": [list(p) for p in incumbents], "routing_seconds": routing_seconds,
            "search_seconds": search_seconds, "candidate_count": len(archive),
            "policy_image_evaluations": sum(r["images"] for r in archive.values())}


def search_evaluator(model, x, labels, teacher, device, batch_size, synchronize):
    def evaluate(policy, cutoff):
        errors, kl_sum, seen = 0, 0.0, 0
        start = last_log = perf_counter()
        with torch.no_grad():
            for offset in range(0, len(x), batch_size):
                logits = model(x[offset:offset + batch_size].to(device), policy).cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite search logits.")
                target = teacher[offset:offset + len(logits)]
                errors += int(logits.argmax(-1).ne(labels[offset:offset + len(logits)]).sum())
                kl_sum += float((target * (target.clamp_min(1e-30).log() - logits.log_softmax(-1))).sum(-1).clamp_min(0).sum())
                seen += len(logits)
                now = perf_counter()
                if seen == len(logits) or seen == len(x) or now - last_log >= 10:
                    LOGGER.info("%s; mistakes=%d, error_cutoff=%s.", progress_message("search evaluation", seen, len(x), start), errors, cutoff)
                    last_log = now
                if cutoff is not None and errors > cutoff:
                    break
        synchronize(device)
        return {"errors": errors, "mean_kl": kl_sum / seen, "images": seen, "complete": seen == len(x)}
    return evaluate


def progress_message(phase, seen, total, start):
    elapsed = perf_counter() - start
    eta = elapsed * (total - seen) / seen if seen else 0
    return f"[{phase}] {seen}/{total} images ({100 * seen / total:.1f}%); elapsed={elapsed:.1f}s, ETA~{eta:.1f}s"


def row_batches(runner, rows, batch_size, phase=None):
    start = last_log = perf_counter()
    for offset in range(0, len(rows), batch_size):
        yield runner.load_images(rows[offset:offset + batch_size])
        seen, now = min(offset + batch_size, len(rows)), perf_counter()
        if phase and (offset == 0 or seen == len(rows) or now - last_log >= 10):
            runner.log(progress_message(phase, seen, len(rows), start))
            last_log = now


def graph_signature(model):
    return hashlib.sha256(str(model).encode()).hexdigest()


def profile_metadata(runner, model, manifest, device, checkpoint_hash):
    data = {}
    for split in ("observer", "profile"):
        data[split] = [{"sha256": file_digest(runner.IMAGE_ROOT / row["file"]), "label": row["label"]} for row in manifest[split]]
    return {"version": VERSION, "checkpoint_sha256": checkpoint_hash, "graph": graph_signature(model),
            "data": data, "transform": repr(runner.TRANSFORM), "torch": str(torch.__version__),
            "profile_batch_size": getattr(runner, "PROFILE_BATCH_SIZE", 4),
            "cuda_runtime": runner.device_runtime(device) if torch.device(device).type == "cuda" else None,
            "device": str(device), "modes": [list(m) for m in MODES], "quantization": QUANTIZATION,
            "score": "clamped_squared_elementwise_gradient_error_sum_CE_per_image_snapshot_v2", "upstream": UPSTREAM}


def prepare_profile(runner, model, manifest, device, checkpoint_hash, cache_path):
    start = perf_counter()
    runner.log(f"[profile] Fingerprinting checkpoint graph, observer/profile images and quantization settings; cache={cache_path}.")
    metadata = profile_metadata(runner, model, manifest, device, checkpoint_hash)
    runner.log(f"[profile] Fingerprint completed in {perf_counter() - start:.2f}s.")
    timings = {"fingerprint_seconds": perf_counter() - start, "calibration_seconds": 0.0, "profiling_seconds": 0.0, "cache_read_seconds": 0.0}
    if cache_path.is_file():
        start = perf_counter()
        runner.log(f"[profile] Validating cached sensitivities, activation bounds and costs: {cache_path}.")
        saved = json.loads(cache_path.read_text())
        legacy_reference_column = False
        if saved["metadata"] != metadata:
            # Removing the old zero-score reference choice does not alter the four local scores.
            legacy_metadata = dict(metadata, modes=[*metadata.get("modes", []), [32, 32]])
            if metadata.get("modes") == [list(m) for m in MODES] and saved["metadata"] == legacy_metadata:
                legacy_reference_column = True
            else:
                raise ValueError("Profile cache fingerprint mismatch. Use a different --profile-cache path or remove the obsolete cache.")
        model.restore_bounds(saved["activation_bounds"])
        if set(saved["operator_macs"]) != set(model.operators):
            raise ValueError("Profile cache operator mismatch.")
        for name, count in saved["operator_macs"].items():
            if type(count) is not int or count <= 0:
                raise ValueError("Invalid cached MAC count.")
            model.operators[name].affine.macs = count
        scores = np.asarray(saved["scores"], dtype=float)
        costs = cost_table(model)
        cached_costs = np.asarray(saved["costs"], dtype=float)
        if legacy_reference_column:
            if (scores.shape != (len(model.operators), len(MODES) + 1)
                    or cached_costs.shape != (*scores.shape, 2)
                    or not np.isfinite(cached_costs).all() or np.any(scores[:, -1] != 0)):
                raise ValueError("Invalid legacy FP32-reference column in cached Fisher table.")
            scores, cached_costs = scores[:, :-1], cached_costs[:, :-1]
            runner.log("[profile] Discarded the legacy FP32 reference column in memory; reusing four quantized-format scores. Original cache is unchanged.")
        if scores.shape != costs.shape[:2] or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("Invalid cached Fisher table.")
        if not np.array_equal(costs, cached_costs):
            raise ValueError("Cached cost table does not match the model.")
        timings["cache_read_seconds"] = perf_counter() - start
        timings["shared_profile_build_seconds"] = sum(saved.get("original_timings", {}).get(k, 0.0) for k in ("calibration_seconds", "profiling_seconds"))
        runner.log(f"[profile] Reused Fisher profile in {timings['cache_read_seconds']:.2f}s: {cache_path}")
        return scores, costs, metadata, timings, True
    start = perf_counter()
    runner.log(f"[calibration] Started: {len(manifest['observer'])} observer images, batch_size={runner.BATCH_SIZE}.")
    calibrate(model, row_batches(runner, manifest["observer"], runner.BATCH_SIZE, "calibration"), device)
    runner.sync(device)
    timings["calibration_seconds"] = perf_counter() - start
    runner.log(f"[calibration] Completed in {timings['calibration_seconds']:.2f}s; activation ranges are now frozen.")
    start = perf_counter()
    profile_batch = getattr(runner, "PROFILE_BATCH_SIZE", 4)
    runner.log(f"[Fisher profiling] Started: {len(manifest['profile'])} labeled images, {len(model.operators)} operators, backward_microbatch={profile_batch}.")
    scores = fisher_scores(model, row_batches(runner, manifest["profile"], profile_batch, "Fisher profiling"), device)
    runner.sync(device)
    timings["profiling_seconds"] = perf_counter() - start
    runner.log(f"[Fisher profiling] Completed in {timings['profiling_seconds']:.2f}s; scoring and cost tables are fixed for search.")
    timings["shared_profile_build_seconds"] = timings["calibration_seconds"] + timings["profiling_seconds"]
    costs = cost_table(model)
    write_json(cache_path, {"metadata": metadata, "activation_bounds": model.bounds(), "scores": scores.tolist(),
                            "costs": costs.tolist(), "operator_macs": {n: m.affine.macs for n, m in model.operators.items()},
                            "original_timings": timings})
    runner.log(f"[profile] Saved reusable calibration, sensitivity and cost tables: {cache_path}.")
    return scores, costs, metadata, timings, False


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_policy(model, policy, path):
    policy = validate_policy(policy, len(model.operators))
    rows = []
    for (name, op), mode in zip(model.operators.items(), policy):
        w, a = MODES[mode]
        for kind, bits in (("weight", w), ("activation_quantizer", a)):
            rows.append({"name": name if kind == "weight" else name + ".input_quantizer", "kind": kind,
                         "weight_bits": w if kind == "weight" else "", "activation_bits": a,
                         "quant_bits": bits, "input_quantizer": name + ".input_quantizer" if kind == "weight" else "",
                         "scale": float(op.input_quantizer.bound.cpu()) / (2 ** (a - 1) - 1) if kind != "weight" else "",
                         "quant_min": -(2 ** (bits - 1) - 1),
                         "quant_max": 2 ** (bits - 1) - 1,
                         "granularity": "per_output_channel" if kind == "weight" else "per_tensor",
                         "parameter_count": op.affine.count if kind == "weight" else "",
                         "compute_dtype": "float32", "bias_bits": 32 if kind == "weight" else "",
                         "notes": "Symmetric signed; ties to even; fake quantization."})
    write_csv(path, rows)


def policy_document(model, policy, checkpoint_hash):
    policy = validate_policy(policy, len(model.operators))
    return {"schema_version": VERSION, "granularity": "operator", "quantization": QUANTIZATION,
            "checkpoint_sha256": checkpoint_hash, "graph": graph_signature(model),
            "assignments": {name: {"weight_bits": MODES[m][0], "activation_bits": MODES[m][1]} for name, m in zip(model.operators, policy)},
            "activation_bounds": model.bounds(), "quantization_rounding": "nearest_ties_to_even"}


def restore_policy(model, saved, checkpoint_hash):
    if saved.get("schema_version") != VERSION or saved.get("granularity") != "operator" or saved.get("quantization") != QUANTIZATION:
        raise ValueError("Unsupported layer policy schema or quantization semantics.")
    if saved.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Saved policy checkpoint hash does not match --checkpoint.")
    if saved.get("graph") != graph_signature(model) or set(saved["assignments"]) != set(model.operators):
        raise ValueError("Saved policy does not match the operator graph.")
    policy = []
    for name in model.operators:
        assignment = saved["assignments"][name]
        bits = (assignment.get("weight_bits"), assignment.get("activation_bits"))
        if bits == (32, 32):
            raise ValueError(f"FP32 assignment for {name} is not a quantized policy format. FP32 is a baseline only; rerun search with the four quantized formats.")
        if any(type(v) is not int for v in bits) or bits not in MODES:
            raise ValueError(f"Unsupported precision for {name}: {bits}")
        policy.append(MODES.index(bits))
    model.restore_bounds(saved["activation_bounds"])
    return tuple(policy)


def heldout_evaluate(runner, model, rows, policies, device, path):
    if not rows:
        raise ValueError("A nonempty held-out test split is required.")
    names = ["FP32", *policies]
    stats = {name: {"correct": 0, "top5_correct": 0, "kl_sum": 0.0} for name in names}
    runner.sync(device)
    start = perf_counter()
    last_log = start
    runner.log(f"[held-out] Started: {len(rows)} images, {len(names)} models (including FP32), batch_size={runner.BATCH_SIZE}; predictions={path}.")
    with Path(path).open("w", newline="") as stream, torch.no_grad():
        writer = csv.writer(stream)
        writer.writerow(["file", "label", *names])
        for offset in range(0, len(rows), runner.BATCH_SIZE):
            batch = rows[offset:offset + runner.BATCH_SIZE]
            x, labels = runner.load_images(batch)
            x = x.to(device)
            reference = model(x).cpu()
            teacher = reference.softmax(-1)
            predictions = []
            for name in names:
                logits = reference if name == "FP32" else model(x, policies[name]).cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError(f"Non-finite held-out logits for {name}, batch starting at image {offset}.")
                pred = logits.argmax(-1)
                predictions.append(pred)
                stats[name]["correct"] += int(pred.eq(labels).sum())
                stats[name]["top5_correct"] += int(logits.topk(min(5, logits.shape[-1]), dim=-1).indices.eq(labels[:, None]).any(-1).sum())
                if name != "FP32":
                    stats[name]["kl_sum"] += float((teacher * (teacher.clamp_min(1e-30).log() - logits.log_softmax(-1))).sum(-1).clamp_min(0).sum())
            writer.writerows([row["file"], int(labels[i]), *[int(p[i]) for p in predictions]] for i, row in enumerate(batch))
            seen, now = min(offset + runner.BATCH_SIZE, len(rows)), perf_counter()
            if offset == 0 or seen == len(rows) or now - last_log >= 10:
                runner.log(progress_message("held-out", seen, len(rows), start) + f"; {seen * len(names)} policy-image evaluations.")
                last_log = now
    runner.sync(device)
    for name, value in stats.items():
        value.update(images=len(rows), top1_percent=100 * value["correct"] / len(rows),
                     top5_percent=100 * value["top5_correct"] / len(rows), mean_kl=value.pop("kl_sum") / len(rows))
        runner.log(f"[held-out] {name}: top1={value['top1_percent']:.3f}%, top5={value['top5_percent']:.3f}%, KL={value['mean_kl']:.6g}.")
    runner.log(f"[held-out] Completed in {perf_counter() - start:.2f}s.")
    return stats, perf_counter() - start


def evaluate_saved(runner, saved, export_only=False):
    torch.set_num_threads(2)
    device = runner.select_device()
    digest = file_digest(runner.CHECKPOINT)
    if digest != saved.get("checkpoint_sha256"):
        raise ValueError("Saved policy checkpoint hash does not match --checkpoint.")
    model = load_model(runner.CHECKPOINT, device)
    policy = restore_policy(model, saved, digest)
    export_policy(model, policy, runner.OUT / "selected_policy.csv")
    write_json(runner.OUT / "selected_policy.json", policy_document(model, policy, digest))
    if not export_only:
        manifest = runner.read_manifest()
        stats, seconds = heldout_evaluate(runner, model, manifest.get("test", []), {"mixed": policy}, device, runner.OUT / "evaluation_predictions.csv")
        write_json(runner.OUT / "evaluation.json", {"models": stats, "seconds": seconds, "device": str(device), "batch_size": runner.BATCH_SIZE})


def frontier(runner, args):
    import matplotlib  # Fail before expensive work if the plotting dependency is absent.
    total_start = perf_counter()
    torch.set_num_threads(2)
    if args.budget_points < 2 or args.max_candidates < 1:
        raise ValueError("Use at least two budgets and a positive candidate allowance.")
    runner.log("[dataset] Validating manifest files, labels and split separation.")
    manifest = runner.read_manifest()
    runner.log("[dataset] Validated split counts: " + json.dumps({k: len(v) for k, v in manifest.items()}))
    if not manifest.get("test"):
        raise ValueError("fisher-frontier requires a held-out test split.")
    device = runner.select_device()
    start = perf_counter()
    runner.log(f"[model] Hashing checkpoint and loading BN-folded ResNet-50 on {device}: {runner.CHECKPOINT}.")
    digest = file_digest(runner.CHECKPOINT)
    model = load_model(runner.CHECKPOINT, device)
    runner.sync(device)
    model_load_seconds = perf_counter() - start
    runner.log(f"[model] Loaded {len(model.operators)} independent operators in {model_load_seconds:.2f}s; weights remain fixed.")
    runner.log("[model] Search formats: " + ", ".join(f"W{w}A{a}" for w, a in MODES) + "; FP32 is a separate baseline only.")
    cache_path = (args.profile_cache or runner.OUT / "fisher_profile.json").expanduser().resolve()
    scores, costs, metadata, timings, cache_hit = prepare_profile(runner, model, manifest, device, digest, cache_path)
    write_csv(runner.OUT / "sensitivity_costs.csv", [
        {"operator": name, "mode": mode, "weight_bits": w, "activation_bits": a,
         "fisher_score": float(scores[i, mode]), "estimated_storage_bytes": int(costs[i, mode, 0]), "bitops": float(costs[i, mode, 1])}
        for i, name in enumerate(model.operators) for mode, (w, a) in enumerate(MODES)])
    runner.log(f"[outputs] Saved sensitivity and cost table: {runner.OUT / 'sensitivity_costs.csv'}.")
    low, high = policy_cost(costs, (0,) * len(scores)), policy_cost(costs, (2,) * len(scores))
    budgets = np.linspace(low[0], high[0], args.budget_points)[:, None]
    budget_metrics = ["estimated_storage_bytes"]
    bitops_budget = None
    if args.compute_fraction is not None:
        bitops_budget = float(args.compute_fraction * high[1])
        budgets = np.column_stack((budgets, np.full(args.budget_points, bitops_budget)))
        budget_metrics.append("bitops")
        if not feasible(low, budgets[0]):
            raise ValueError("Explicit BitOps cap is below the minimum all-W4A8 cost; increase --compute-fraction to at least 0.5 or omit it for no cap.")
    runner.log("Storage budgets with " + (f"explicit BitOps cap {bitops_budget:.0f}." if bitops_budget is not None else "no BitOps ceiling; compute costs are reported only."))
    seeds, solver_reports = [], []
    for slot, budget in enumerate(budgets):
        runner.log(f"[MILP] Solving B{slot + 1}/{len(budgets)}: storage ceiling={budget[0] / 1e6:.3f} MB; time limit=10s.")
        seed, report = solve_seed(scores, costs, budget)
        solver_reports.append(report)
        runner.log(f"[MILP] B{slot + 1}: status={report['status']}, optimal={report['optimal']}, seconds={report['seconds']:.2f}; {report['message']}")
        if seed is None:
            write_json(runner.OUT / "solver.json", solver_reports)
            raise RuntimeError(f"No verified feasible ILP incumbent: {report}")
        seeds.append(seed)
        runner.log(f"[MILP] B{slot + 1}: verified feasible, Fisher={report['score']:.6g}, actual storage={report['actual_cost'][0] / 1e6:.3f} MB, BitOps={report['actual_cost'][1] / 1e9:.3f} G.")
        if not report["optimal"]:
            LOGGER.warning("[MILP] B%d uses a verified feasible incumbent without an optimality certificate.", slot + 1)
    write_json(runner.OUT / "solver.json", solver_reports)
    runner.log(f"[MILP] {len(budgets)} budgets produced {len(set(seeds))} unique seeds; duplicate policies will be evaluated once.")
    if len(set(seeds)) > args.max_candidates:
        raise ValueError(f"--max-candidates must cover {len(set(seeds))} unique ILP seeds.")
    start = perf_counter()
    runner.log(f"[search reference] Loading {len(manifest['search'])} search images and computing FP32 teacher outputs; batch_size={runner.BATCH_SIZE}.")
    x, labels = runner.load_images(manifest["search"])
    teacher = runner.batch_forward(model, x, None, device).softmax(-1)
    teacher_seconds = perf_counter() - start
    runner.log(f"[search reference] Ready in {teacher_seconds:.2f}s; held-out labels are not used for routing.")
    evaluator = search_evaluator(model, x, labels, teacher, device, runner.BATCH_SIZE, runner.sync)
    with (runner.OUT / "evaluation_history.jsonl").open("w") as history:
        def report(record):
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
        runner.log(f"[search] Starting mode={args.router_mode}, allowance={args.max_candidates} unique candidates (including seeds and early rejections).")
        search = run_search(scores, costs, budgets, seeds, evaluator, args.router_mode, args.max_candidates, report)
    write_json(runner.OUT / "search.json", {**search, "budgets": budgets.tolist(), "budget_metrics": budget_metrics,
                                           "bitops_budget": bitops_budget, "router_mode": args.router_mode})
    write_json(runner.OUT / "transfers.json", search["transfers"])
    front_policies = search_frontier_policies(search["records"], 0)
    bitops_front_policies = search_frontier_policies(search["records"], 1)
    selected = sorted(front_policies | bitops_front_policies | set(seeds), key=lambda p: (policy_cost(costs, p)[0], p))
    runner.log(f"[selection] Search frontiers: storage={len(front_policies)}, BitOps={len(bitops_front_policies)}; union plus seeds={len(selected)} policies.")
    named = {f"P{i + 1:02}": p for i, p in enumerate(selected)}
    # Persist the frozen shortlist before any held-out inference or accuracy calculation.
    write_json(runner.OUT / "selection.json", {"policies": {k: list(p) for k, p in named.items()}, "source": "search split only"})
    for name, policy in named.items():
        write_json(runner.OUT / "policies" / f"{name}.json", policy_document(model, policy, digest))
        export_policy(model, policy, runner.OUT / "policies" / f"{name}.csv")
    del x, labels, teacher, evaluator
    runner.log(f"Selection frozen. Evaluating {len(named)} policies plus FP32 on held-out data.")
    measured, final_seconds = heldout_evaluate(runner, model, manifest["test"], named, device, runner.OUT / "frontier_predictions.csv")
    floating_cost = fp32_reference_cost(model)
    rows = []
    for name, policy in [("FP32", None), *named.items()]:
        cost = floating_cost if policy is None else policy_cost(costs, policy)
        rows.append({"id": name, "storage_bytes": int(cost[0]), "compression_ratio": float(floating_cost[0] / cost[0]),
                     **measured[name], "bitops": float(cost[1]), "bitops_vs_w8a8": float(cost[1] / high[1]),
                     "accuracy_drop_pp": float(measured["FP32"]["top1_percent"] - measured[name]["top1_percent"]),
                     "within_bitops_budget": bool(bitops_budget is None or cost[1] <= bitops_budget + 1e-9),
                     "search_frontier": policy in front_policies, "search_bitops_frontier": policy in bitops_front_policies,
                     "ilp_seed": policy in seeds, "validation_frontier": False, "validation_bitops_frontier": False,
                     "policy_file": "" if policy is None else f"policies/{name}.json"})
    eligible = [i for i, row in enumerate(rows) if row["id"] != "FP32" and row["within_bitops_budget"]]
    for cost_key, flag in (("storage_bytes", "validation_frontier"), ("bitops", "validation_bitops_frontier")):
        for i in runner.pareto_indices([rows[j][cost_key] for j in eligible], [-rows[j]["correct"] for j in eligible]):
            rows[eligible[i]][flag] = True
    write_csv(runner.OUT / "frontier.csv", rows)
    runner.log(f"[outputs] Saved measured frontier table: {runner.OUT / 'frontier.csv'}.")
    timings.update(model_load_seconds=model_load_seconds, teacher_search_seconds=teacher_seconds,
                   solver_seconds=sum(r["seconds"] for r in solver_reports), routing_seconds=search["routing_seconds"],
                   search_seconds=search["search_seconds"], final_evaluation_seconds=final_seconds)
    plot_note = ("No BitOps ceiling; FP32 is a baseline only, outside the quantized frontier." if bitops_budget is None else
                 "Explicit BitOps cap applies to quantized policies; FP32 is a baseline only.")
    runner.log("[plots] Generating accuracy versus storage compression and accuracy versus BitOps graphs.")
    runner.plot_frontier(rows, runner.OUT / "frontier.png", "ResNet-50: Fisher scoring + " + args.router_mode,
                         "Measured shortlist only; estimated storage and BitOps; fake quantization.\n" + plot_note)
    runner.plot_frontier(rows, runner.OUT / "frontier_bitops.png", "ResNet-50: Fisher scoring + " + args.router_mode,
                         "Measured shortlist only; 1 GigaBitOps = 10^9 MAC-bit products, not measured latency.\n" + plot_note,
                         flag="validation_bitops_frontier", x_metric="bitops")
    runner.log(f"[plots] Saved PNG/SVG graphs: {runner.OUT / 'frontier.png'} and {runner.OUT / 'frontier_bitops.png'}.")
    timings["total_seconds"] = perf_counter() - total_start
    timings["total_seconds_charging_shared_profile"] = timings["total_seconds"] + (timings["shared_profile_build_seconds"] if cache_hit else 0.0)
    summary = {"router_mode": args.router_mode, "profile_cache_hit": cache_hit, "profile_cache": str(cache_path),
               "quantization_formats": [list(m) for m in MODES], "fp32_baseline_only": True,
               "profile_fingerprint": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
               "device": str(device), "torch_version": str(torch.__version__), "batch_size": runner.BATCH_SIZE,
               "profile_batch_size": getattr(runner, "PROFILE_BATCH_SIZE", 4), "runtime": runner.device_runtime(device),
               "operators": len(scores), "candidate_limit": args.max_candidates, "candidate_count": search["candidate_count"],
               "policy_image_evaluations": search["policy_image_evaluations"],
               "transfer_evaluations": sum(r["kind"] == "transfer" for r in search["records"]),
               "successful_edits": len(search["transfers"]), "budgets": budgets.tolist(), "budget_metrics": budget_metrics,
               "compute_fraction": args.compute_fraction, "bitops_budget": bitops_budget,
               "split_counts": {k: len(v) for k, v in manifest.items()}, "timings": timings, "candidates": rows,
               "split_class_counts": {k: len({row['label'] for row in v}) for k, v in manifest.items()},
               "scope": "Measured storage and BitOps frontiers among shortlisted policies, not global optima. Estimated storage excludes container overhead; BitOps excludes float residual/requantization work. No integer runtime or speedup claim.",
               "quantization": QUANTIZATION}
    write_json(runner.OUT / "frontier.json", summary)
    runner.log(f"[outputs] Saved run summary: {runner.OUT / 'frontier.json'}; history: {runner.OUT / 'evaluation_history.jsonl'}; policies: {runner.OUT / 'policies'}.")
    runner.log(json.dumps({k: summary[k] for k in ("candidate_count", "transfer_evaluations", "policy_image_evaluations", "timings")}, indent=2))
