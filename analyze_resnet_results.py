"""Audit saved results and compute paired, class-stratified uncertainty."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

OUT = Path(__file__).resolve().parent / "resnet_results"
c = json.loads((OUT / "comparison.json").read_text())
s = json.loads((OUT / "summary.json").read_text())
o = np.load(OUT / "grid_oracle.npz")
index = {tuple(p): i for i, p in enumerate(o["policies"])}
for grid in c["grid"]:
    feasible = (o["bytes"] <= grid["memory_fraction"] * s["resource"]["all_w8a8_bytes"]) & (o["bitops"] <= grid["compute_fraction"] * s["resource"]["all_w8a8_bitops"])
    ids = np.flatnonzero(feasible)
    assert len(ids) == grid["feasible"]
    exact = min(ids, key=lambda i: (o["per_image"][i, :, 2].sum(), o["per_image"][i, :, 0].mean()))
    assert o["policies"][exact].tolist() == grid["exact_policy"]
    certificates = grid["race"]["certificate"]
    assert {tuple(row["policy"]) for row in certificates} == {tuple(o["policies"][i]) for i in ids}
    for row in certificates:
        total = int(o["per_image"][index[tuple(row["policy"])] , :, 2].sum())
        assert 0 <= row["mistakes_seen"] <= min(total, row["images_seen"])
        assert row["complete"] == (row["images_seen"] == 100)
        if row["complete"]:
            assert row["mistakes_seen"] == total
        else:
            assert row["mistakes_seen"] > grid["race"]["initial_total_errors"] >= grid["exact_errors"]

manifest = json.loads((OUT / "manifest.json").read_text())
all_hashes = [row["sha256"] for rows in manifest.values() for row in rows]
assert len(all_hashes) == len(set(all_hashes)) == 750
for split, rows in manifest.items():
    assert all(("ILSVRC2012_val_" in row["file"]) == (split == "test") for row in rows)

chosen = np.load(OUT / "test_2_2_0_0.npz")
stats = {}
for name, filename in [("float", "test_float.npz"), ("all_w4a8", "test_0_0_0_0.npz")]:
    baseline = np.load(OUT / filename)
    assert np.array_equal(chosen["labels"], baseline["labels"])
    good = chosen["predicted"] == chosen["labels"]
    reference = baseline["predicted"] == baseline["labels"]
    delta = good.astype(int) - reference.astype(int)
    rng = np.random.default_rng(20261111)
    bootstrap = np.zeros(20000)
    for label in np.unique(chosen["labels"]):
        group = delta[chosen["labels"] == label]
        bootstrap += rng.choice(group, size=(20000, len(group)), replace=True).sum(1)
    bootstrap *= 100 / len(delta)
    wins, losses = int((delta == 1).sum()), int((delta == -1).sum())
    stats[name] = {"selected_accuracy_percent": 100 * float(good.mean()), "baseline_accuracy_percent": 100 * float(reference.mean()), "difference_percentage_points": 100 * float(delta.mean()), "paired_class_stratified_bootstrap_95_percent_interval_pp": np.quantile(bootstrap, [.025, .975]).tolist(), "selected_only_correct": wins, "baseline_only_correct": losses, "exact_mcnemar_p": float(binomtest(wins, wins + losses, .5).pvalue)}
timing = {key: {"median": float(np.median([r[key] for r in c["timing_repetitions"]])), "min": min(r[key] for r in c["timing_repetitions"]), "max": max(r[key] for r in c["timing_repetitions"])} for key in c["timing_repetitions"][0]}
seed = json.loads((OUT / "seed_timing.json").read_text())["seconds"]
timing["seed_seconds"] = {"median": float(np.median(seed)), "min": min(seed), "max": max(seed)}
timing["race_plus_seed_estimated_seconds"] = timing["race_seconds"]["median"] + timing["seed_seconds"]["median"]
result = {"certificate_audit": "All nine budget certificates consistent with exhaustive oracle; no split duplicates or original-validation leakage.", "accuracy_statistics": stats, "timing": timing, "accuracy_optimum_hits_of_9": {name: sum(g["methods"][name]["error_regret"] == 0 for g in c["grid"]) for name in c["grid"][0]["methods"]}}
(OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
