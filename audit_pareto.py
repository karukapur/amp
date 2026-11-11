"""Independent checks of the retained Pareto pilot; writes only runs/pareto_audit.

Run: .venv/bin/python audit_pareto.py
The independent model uses torchvision's graph, manual BN folding, packed integer
weights reloaded from disk, and hooks implementing the documented quantizers.
It does not use QuantResNet.forward, traverse, losses, or costs for its inference.
This is a storage round trip plus FP32 simulation, not an integer-kernel export.
"""
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import binomtest
from torchvision.models import resnet50

import resnet_benchmark as runner

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/pareto_pilot"
OUT = ROOT / "runs/pareto_audit"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [100 * (center-half), 100 * (center+half)]


def audit_data_and_statistics():
    manifest = json.loads(runner.MANIFEST.read_text())
    mapping = json.loads((runner.DATA / "imagenet_class_index.json").read_text())
    hashes, pixels, sizes = [], [], Counter()
    for split, records in manifest.items():
        for row in records:
            path = ROOT / row["file"]
            assert digest(path) == row["sha256"]
            assert mapping[str(row["label"])][0] == row["wnid"]
            assert Path(row["archive_member"]).parent.name == row["wnid"]
            assert path.name == Path(row["archive_member"]).name
            assert path.name.startswith("ILSVRC2012_val_") == (split == "test")
            hashes.append(row["sha256"])
            with Image.open(path) as im:
                im = im.convert("RGB")
                pixels.append(hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest())
                sizes[min(im.size)] += 1
    assert len(set(hashes)) == len(hashes) == 750
    assert len(set(pixels)) == len(pixels)
    saved = list(csv.DictReader((SOURCE / "frontier_predictions.csv").open()))
    assert [r["file"] for r in saved] == [r["file"] for r in manifest["test"]]
    y = np.array([r["label"] for r in manifest["test"]])
    assert np.array_equal(y, [int(r["label"]) for r in saved])
    summary = json.loads((SOURCE / "frontier.json").read_text())
    predictions = {name: np.array([int(r[name]) for r in saved]) for name in saved[0] if name not in ("file", "label")}
    for row in summary["candidates"]:
        assert int(sum(predictions[row["id"]] == y)) == row["correct"]
        assert row["top1_percent"] == 100 * row["correct"] / len(y)
    base, mixed = predictions["FP32"] == y, predictions["P04"] == y
    difference = mixed.astype(int) - base.astype(int)
    rng = np.random.default_rng(20261111)
    bootstrap = np.zeros(20000)
    class_differences = []
    per_class = []
    for label in sorted(set(y)):
        d = difference[y == label]
        bootstrap += d[rng.integers(len(d), size=(20000, len(d)))].sum(1) / len(y) * 100
        class_differences.append(float(d.mean() * 100))
        per_class.append({"label": int(label), "class": mapping[str(label)][1], "images": int(sum(y == label)),
                          "FP32_correct": int(base[y == label].sum()), "P04_correct": int(mixed[y == label].sum())})
    # Sensitivity only: these ten deliberately chosen classes are not a random
    # sample of ImageNet classes, so this is not an ImageNet confidence interval.
    clustered = np.array(class_differences)[rng.integers(10, size=(20000, 10))].mean(1)
    stats = {
        "scope": "Intervals condition on this selected ten-class distribution; post-selection exploratory inference.",
        "images": len(y), "FP32_correct": int(base.sum()), "P04_correct": int(mixed.sum()),
        "FP32_wilson_95_percent": wilson(int(base.sum()), len(y)),
        "P04_wilson_95_percent": wilson(int(mixed.sum()), len(y)),
        "difference_pp": float(difference.mean()*100),
        "paired_class_stratified_bootstrap_95_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
        "class_resampling_sensitivity_95_pp": np.quantile(clustered, [.025, .975]).tolist(),
        "P04_only_correct": int(sum(mixed & ~base)), "FP32_only_correct": int(sum(base & ~mixed)),
        "exact_mcnemar_two_sided_p": binomtest(int(sum(mixed & ~base)), int(sum(base != mixed))).pvalue,
        "per_class": per_class,
    }
    write_json("statistics.json", stats)
    write_json("data_integrity.json", {"checkpoint_sha256": digest(runner.CHECKPOINT),
        "manifest_sha256": digest(runner.MANIFEST), "runner_sha256": digest(Path(runner.__file__)),
        "all_image_hashes_match": True, "unique_encoded_images": len(set(hashes)),
        "unique_decoded_images": len(set(pixels)), "short_side_pixels": dict(sizes),
        "split_counts": {k: len(v) for k, v in manifest.items()},
        "classes_per_split": {k: dict(Counter(r["label"] for r in v)) for k, v in manifest.items()},
        "original_validation_excluded_from_calibration_profile_search": True,
        "all_saved_prediction_counts_match": True})
    print("Data and statistics:", json.dumps(stats), flush=True)
    return manifest, predictions


def audit_search():
    z = np.load(SOURCE / "frontier_search.npz")
    policies, sizes, metrics = z["policies"], z["bytes"], z["per_image"]
    assert metrics.shape == (256, 100, 3) and np.isfinite(metrics).all()
    assert set(map(tuple, policies)) == set(__import__("itertools").product(range(4), repeat=4))
    assert set(np.unique(metrics[:, :, 2])) == {0, 1}
    errors, kl = metrics[:, :, 2].sum(1), metrics[:, :, 0].mean(1)
    # Deliberately independent of frontier_candidates/pareto_indices.
    winners = [min(np.flatnonzero(sizes == s), key=lambda i: (errors[i], kl[i], z["bitops"][i], i)) for s in sorted(set(sizes))]
    front = [i for i in winners if not any(sizes[j] <= sizes[i] and errors[j] <= errors[i]
             and (sizes[j] < sizes[i] or errors[j] < errors[i]) for j in winners)]
    summary = json.loads((SOURCE / "frontier.json").read_text())
    recorded = [tuple(map(int, row["policy"].split(","))) for row in summary["candidates"] if row["search_frontier"]]
    assert set(recorded) == set(map(tuple, policies[front]))
    p04 = int(np.flatnonzero((policies == [2, 3, 0, 1]).all(1))[0])
    same_size = np.flatnonzero(sizes == sizes[p04])
    rng = np.random.default_rng(20261111)
    selected = Counter()
    labels = np.array([r["label"] for r in json.loads(runner.MANIFEST.read_text())["search"]])
    for _ in range(2000):
        idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), sum(labels == c)) for c in sorted(set(labels))])
        e = metrics[same_size][:, idx, 2].sum(1)
        k = metrics[same_size][:, idx, 0].mean(1)
        winner = min(range(len(same_size)), key=lambda j: (e[j], k[j], z["bitops"][same_size[j]], same_size[j]))
        selected[",".join(map(str, policies[same_size[winner]]))] += 1
    write_json("search_audit.json", {"all_256_policies_present": True, "independent_frontier_matches": True,
        "search_frontier": policies[front].tolist(), "search_correct": (100-errors[front]).astype(int).tolist(),
        "P04_search_correct": int(100-errors[p04]), "policies_at_P04_storage": len(same_size),
        "P04_storage_bootstrap_winners_2000": dict(selected),
        "bootstrap_note": "Search-label resampling stability, not independent generalization evidence."})


def manual_fold(conv, bn):
    w = conv.weight.detach()
    factor = bn.weight.detach() * torch.rsqrt(bn.running_var + bn.eps)
    bias = torch.zeros_like(bn.running_mean) if conv.bias is None else conv.bias.detach()
    conv.weight = torch.nn.Parameter(w * factor[:, None, None, None], requires_grad=False)
    conv.bias = torch.nn.Parameter((bias-bn.running_mean) * torch.rsqrt(bn.running_var+bn.eps) * bn.weight.detach() + bn.bias.detach(), requires_grad=False)


def independent_model(original, saved):
    """Return torchvision graph with independent packing and quantization hooks."""
    model = copy.deepcopy(original)
    manual_fold(model.conv1, model.bn1)
    model.bn1 = torch.nn.Identity()
    pairs = [("conv1", "stem_op", model.conv1, 8), ("fc", "head_op", model.fc, 8)]
    acts = [(model.conv1, "input", 8), (model.fc, "head_input", 8)]
    for s in range(4):
        wb, ab = ((4, 8), (4, 16), (8, 8), (8, 16))[saved["policy"][s]]
        for b, block in enumerate(getattr(model, f"layer{s+1}")):
            prefix = f"stages.{s}.blocks.{b}"
            acts.extend([(block, prefix+".input", ab), (block.conv2, prefix+".a2", ab), (block.conv3, prefix+".a3", ab)])
            for c in range(1, 4):
                conv = getattr(block, f"conv{c}")
                manual_fold(conv, getattr(block, f"bn{c}"))
                setattr(block, f"bn{c}", torch.nn.Identity())
                pairs.append((f"layer{s+1}.{b}.conv{c}", prefix+f".c{c}", conv, wb))
            if block.downsample is not None:
                manual_fold(block.downsample[0], block.downsample[1])
                block.downsample[1] = torch.nn.Identity()
                pairs.append((f"layer{s+1}.{b}.downsample.0", prefix+".projection", block.downsample[0], wb))
    metadata = []
    weight_bytes = scale_bytes = bias_bytes = 0
    packed_path = OUT / "P04_packed_payload.bin"
    with packed_path.open("wb") as f:
        for name, runner_name, layer, bits in pairs:
            weight = layer.weight.detach().numpy()
            maximum = np.max(np.abs(weight).reshape(weight.shape[0], -1), axis=1)
            scale = np.maximum(maximum, np.float32(1e-10)) / np.float32(2**(bits-1)-1)
            scale_shape = (len(scale),) + (1,)*(weight.ndim-1)
            integers = np.clip(np.rint(weight / scale.reshape(scale_shape)), -(2**(bits-1)-1), 2**(bits-1)-1).astype(np.int8).ravel()
            if bits == 4:
                # Offset-binary nibbles; two weights per byte, reversible including -7.
                values = (integers.astype(np.int16)+8).astype(np.uint8)
                if len(values) % 2:
                    values = np.append(values, np.uint8(8))
                packed = (values[::2] | (values[1::2] << 4)).tobytes()
            else:
                packed = integers.tobytes()
            bias = layer.bias.detach().numpy().astype("<f4").tobytes()
            scale_data = scale.astype("<f4").tobytes()
            metadata.append({"name": name, "runner_name": runner_name, "bits": bits, "shape": list(weight.shape),
                             "offset": f.tell(), "weight_bytes": len(packed), "scale_bytes": len(scale_data), "bias_bytes": len(bias)})
            f.write(packed + scale_data + bias)
            weight_bytes += len(packed)
            scale_bytes += len(scale_data)
            bias_bytes += len(bias)
        # One FP32 bound per activation quantizer, equivalent in size to a scale.
        assert len(acts) == 50 and {a[1] for a in acts} == saved["activation_bounds"].keys()
        f.write(np.array([saved["activation_bounds"][name] for _, name, _ in acts], dtype="<f4").tobytes())
    with packed_path.open("rb") as f:
        for entry, (_, _, layer, bits) in zip(metadata, pairs):
            assert f.tell() == entry["offset"]
            data = np.frombuffer(f.read(entry["weight_bytes"]), dtype=np.uint8)
            if bits == 4:
                integer = np.stack((data & 15, data >> 4), axis=1).ravel().astype(np.int16)-8
                integer = integer[:int(np.prod(entry["shape"]))]
            else:
                integer = data.view(np.int8)
            scale = np.frombuffer(f.read(entry["scale_bytes"]), dtype="<f4").copy()
            restored = integer.reshape(entry["shape"]).astype(np.float32) * scale.reshape((-1,) + (1,)*(len(entry["shape"])-1))
            layer.weight.data.copy_(torch.from_numpy(restored))
            bias = np.frombuffer(f.read(entry["bias_bytes"]), dtype="<f4").copy()
            layer.bias.data.copy_(torch.from_numpy(bias))
        bounds = np.frombuffer(f.read(4*len(acts)), dtype="<f4").copy()
        assert not f.read(1)
    for (module, _, bits), bound in zip(acts, bounds):
        qmax = 2**(bits-1)-1
        scale = torch.tensor(float(bound), dtype=torch.float32) / qmax
        def hook(_module, args, scale=scale, qmax=qmax):
            # Clamp integer codes instead of input; same representable grid.
            return (torch.clamp(torch.round(args[0]/scale), -qmax, qmax)*scale,)
        module.register_forward_pre_hook(hook)
    assert packed_path.stat().st_size == saved["packed_bytes"]
    write_json("packing.json", {"payload_bytes": packed_path.stat().st_size,
        "payload_sha256": digest(packed_path), "packed_weight_bytes": weight_bytes,
        "weight_scale_bytes": scale_bytes, "fp32_bias_bytes": bias_bytes, "activation_bound_bytes": 4*len(acts),
        "layers": metadata, "activation_order": [name for _, name, _ in acts],
        "metadata_bytes_excluded_from_estimate": len(json.dumps(metadata).encode()),
        "format": "Raw packed weights, FP32 scales and biases per layer, then 50 FP32 bounds. Layout supplied separately; no integer execution."})
    return model, pairs


def independent_images(rows):
    output = []
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    for row in rows:
        with Image.open(ROOT / row["file"]) as im:
            im = im.convert("RGB")
            w, h = im.size
            size = (256, int(256*h/w)) if w <= h else (int(256*w/h), 256)
            im = im.resize(size, Image.Resampling.BILINEAR)
            left, top = round((im.width-224)/2), round((im.height-224)/2)
            im = im.crop((left, top, left+224, top+224))
            x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float()/255
            output.append((x-mean)/std)
    return torch.stack(output)


def audit_inference(manifest, saved_predictions):
    torch.set_num_threads(2)
    saved = json.loads((SOURCE / "policies/P04.json").read_text())
    assert digest(runner.CHECKPOINT) == saved["checkpoint_sha256"]
    original = resnet50(weights=None).eval().requires_grad_(False)
    original.load_state_dict(torch.load(runner.CHECKPOINT, map_location="cpu", weights_only=True))
    independent, pairs = independent_model(original, saved)
    model = runner.QuantResNet(copy.deepcopy(original)).eval()
    for name, module in model.named_modules():
        if isinstance(module, runner.Activation):
            module.bound.fill_(saved["activation_bounds"][name])
    modules = dict(model.named_modules())
    weight_error = 0.0
    for _, name, layer, bits in pairs:
        err = float((layer.weight - getattr(modules[name], f"w{bits}")).abs().max())
        weight_error = max(weight_error, err)
        assert err < 1e-6, (name, err)
    state_bytes = sum(v.numel()*v.element_size() for v in model.state_dict().values())
    stats = {"device": "cpu", "torch_version": str(torch.__version__), "batch_size": 7,
             "independent_vs_runner_max_weight_error": weight_error, "runner_state_tensor_bytes": state_bytes,
             "original_vs_folded_max_logit_error": 0.0, "independent_vs_runner_max_logit_error": 0.0,
             "preprocess_max_error": 0.0, "logit_shape": [7, 1000]}
    predictions = {key: [] for key in ["original", "folded", "independent_packed_P04", "runner_P04"]}
    retained_logits = []
    with torch.inference_mode():
        for offset in range(0, len(manifest["test"]), 7):
            rows = manifest["test"][offset:offset+7]
            x = independent_images(rows)
            runner_x, _ = runner.load_images(rows)
            stats["preprocess_max_error"] = max(stats["preprocess_max_error"], float((x-runner_x).abs().max()))
            assert torch.equal(x, runner_x)
            logits = [original(x), model(x), independent(x), model(x, saved["policy"])]
            retained_logits.append(logits[3].numpy())
            assert all(t.shape == (len(rows), 1000) and torch.isfinite(t).all() for t in logits)
            stats["original_vs_folded_max_logit_error"] = max(stats["original_vs_folded_max_logit_error"], float((logits[0]-logits[1]).abs().max()))
            stats["independent_vs_runner_max_logit_error"] = max(stats["independent_vs_runner_max_logit_error"], float((logits[2]-logits[3]).abs().max()))
            for key, values in zip(predictions, logits):
                predictions[key].extend(values.argmax(1).tolist())
            if offset % 70 == 0 or offset+7 >= len(manifest["test"]):
                print(f"Independent CPU audit {min(offset+7, len(manifest['test']))}/500", flush=True)
    labels = np.array([r["label"] for r in manifest["test"]])
    for name, values in predictions.items():
        baseline = saved_predictions["FP32" if name in ("original", "folded") else "P04"]
        stats[name] = {"correct": int(sum(np.array(values) == labels)), "saved_prediction_mismatches": int(sum(np.array(values) != baseline))}
    write_json("independent_inference.json", stats)
    np.savez(OUT / "cpu_P04_logits.npz", logits=np.concatenate(retained_logits))
    with (OUT / "independent_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", *predictions])
        for i, row in enumerate(manifest["test"]):
            writer.writerow([row["file"], row["label"], *[v[i] for v in predictions.values()]])
    print("Independent inference:", json.dumps(stats), flush=True)
    assert predictions["original"] == predictions["folded"] == saved_predictions["FP32"].tolist()
    assert predictions["independent_packed_P04"] == predictions["runner_P04"]
    assert stats["original_vs_folded_max_logit_error"] < 2e-4
    assert stats["independent_vs_runner_max_logit_error"] < 2e-4
    assert all(not m.observe for m in model.modules() if isinstance(m, runner.Activation))


def summarize_followups():
    """Summarize completed backend and frontier reruns without new selection."""
    if not (OUT / "backend_fixed/mps_P04_logits.npz").exists():
        return
    from scipy.stats import binomtest
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = list(csv.DictReader((SOURCE / "frontier_predictions.csv").open()))
    labels = np.array([int(r["label"]) for r in original])
    base = np.array([int(r["FP32"]) for r in original]) == labels
    predictions = {
        "Original policy, original MPS": np.array([int(r["P04"]) for r in original]),
        "Original policy, CPU": np.load(OUT / "cpu_P04_logits.npz")["logits"].argmax(1),
        "Original policy, corrected MPS": np.load(OUT / "backend_fixed/mps_P04_logits.npz")["logits"].argmax(1),
    }
    corrected = OUT / "corrected_mps_batch8/frontier_predictions.csv"
    if not corrected.exists():
        corrected = OUT / "corrected_mps/frontier_predictions.csv"
    rows = list(csv.DictReader(corrected.open()))
    assert [r["file"] for r in rows] == [r["file"] for r in original]
    predictions["Reselected policy, corrected MPS"] = np.array([int(r["P04"]) for r in rows])
    cpu_path = OUT / "corrected_selected_cpu/evaluation_predictions.csv"
    if (cpu_path.parent / "evaluation.json").exists():
        cpu_rows = list(csv.DictReader(cpu_path.open()))
        assert [r["file"] for r in cpu_rows] == [r["file"] for r in original]
        predictions["Reselected policy, CPU"] = np.array([int(r["mixed_prediction"]) for r in cpu_rows])
    comparison = {}
    for name, values in predictions.items():
        correct = values == labels
        difference = correct.astype(int) - base.astype(int)
        rng = np.random.default_rng(20261111)
        boot = np.zeros(20000)
        for label in sorted(set(labels)):
            v = difference[labels == label]
            boot += v[rng.integers(len(v), size=(20000, len(v)))].sum(1) / len(labels) * 100
        n = int(sum(correct != base))
        comparison[name] = {"correct": int(sum(correct)), "accuracy_percent": 100*float(correct.mean()),
            "difference_pp": 100*float(difference.mean()),
            "paired_class_stratified_bootstrap_95_pp": np.quantile(boot, [.025, .975]).tolist(),
            "candidate_only_correct": int(sum(correct & ~base)), "FP32_only_correct": int(sum(base & ~correct)),
            "mcnemar_two_sided_p": binomtest(int(sum(correct & ~base)), n).pvalue}
    write_json("accuracy_comparisons.json", comparison)
    names = ["FP32 reference", *comparison]
    counts = [int(sum(base)), *[v["correct"] for v in comparison.values()]]
    accuracy = np.array(counts)/len(labels)*100
    intervals = np.array([wilson(k, len(labels)) for k in counts])
    fig, ax = plt.subplots(figsize=(11, 5.8))
    positions = np.arange(len(names))
    ax.errorbar(accuracy, positions, xerr=np.stack((accuracy-intervals[:, 0], intervals[:, 1]-accuracy)),
                fmt="o", capsize=4, color="#25635a")
    ax.axvline(73, linestyle=":", color="#707070")
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.set_ylim(len(names)-.25, -.65)
    for i, value in enumerate(accuracy):
        ax.annotate(f"{value:.1f}%", (value, i), xytext=(0, -16), textcoords="offset points", ha="center")
    ax.set_xlim(66, 82)
    ax.set_xlabel("Top-1 accuracy on the same 500 images (%)")
    ax.set_title("The 6.94× storage claim survives; the 75.2% headline is not robust", loc="left", pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.2)
    fig.text(.025, .025, "Bars: nominal 95% Wilson intervals; exploratory, conditional on 10 selected classes.\n"
             "All quantized rows: 14,713,544-byte packed payload. FP32: 102,121,888 bytes. No integer speed/memory claim.", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(OUT / "audit_summary.png", dpi=180)
    fig.savefig(OUT / "audit_summary.svg")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, predictions = audit_data_and_statistics()
    audit_search()
    audit_inference(manifest, predictions)
    summarize_followups()
