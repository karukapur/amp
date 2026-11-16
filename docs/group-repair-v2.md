# Group repair v2

V2 lives in new files. Run `resnet_benchmark_v2.py` instead of
`resnet_benchmark.py`; the command arguments are preserved. Use a fresh
`--output-dir` for every run. V2 rejects nonempty output directories before
opening their logs, and never rewrites an existing profile cache.

## Choose precision formats

Edit **only `MODES` near the top of `amp/fisher_router_v2.py`**. All four formats
ship enabled. For the immediate W4A8/W8A8 experiment, comment out the A16 entries:

```python
MODES = (
    (4, 8),
    # (4, 16),
    (8, 8),
    # (8, 16),
)
```

Uncomment them to enable W4A16 or W8A16. A16 denotes integer activation
quantization. Any nonempty subset or order of these four options is supported.
FP32 remains a separate reference. This setting controls the operator-level
`fisher-frontier`, `evaluate`, and `export` paths; the copied legacy stage
commands retain their historical format definitions.

Budget endpoints follow the enabled weight widths. If every enabled format has
the same weight width, the requested storage ceilings collapse to one distinct
budget. A single enabled format produces only its seed policy. The optional
`--compute-fraction` denominator remains all-W8A8 BitOps, even if W8A8 is disabled.

Compatible caches can be reused across subsets and reordered formats: columns
are selected by their weight/activation bit pair in memory. Enabling a format
absent from the cache requires a fresh cache path. Other fingerprint mismatches
also require a fresh cache. Existing caches are never modified.

## Run the experiment

For an existing command, change the script filename and use a fresh output path:

```sh
python resnet_benchmark_v2.py fisher-frontier \
  --imagenet-root /datasets/imagenet \
  --checkpoint /models/resnet50-0676ba61.pth \
  --device cuda:0 --batch-size 32 --profile-batch-size 4 \
  --observer-images 1024 --profile-images 128 --search-images 512 \
  --router-mode transfer --budget-points 8 --max-candidates 64 \
  --output-dir runs/v2_group_64_trial1
```

In v2 the existing mode names select:

| CLI mode | Algorithm recorded in results | Behavior |
|---|---|---|
| `ilp` | `ilp_v2` | Evaluate unique MILP seeds only |
| `local` | `local_v2` | One-/two-layer refinement |
| `transfer` | `group_repair_v2` | Joint assignments within groups of at most five layers |

The `transfer` name preserves existing commands; v2 does not borrow successful
edits between budgets. Measurements and deduplication are shared across budgets.

After selecting the formats, this matched sweep tests 32, 64, and 128 candidates
with the same checkpoint, splits, Fisher cache, and resource definitions:

```sh
# Choose a new RUN_ROOT for each independent trial or format configuration.
RUN_ROOT=runs/v2_w4a8_w8a8_trial1
for allowance in 32 64 128; do
  for mode in ilp local transfer; do
    python resnet_benchmark_v2.py fisher-frontier \
      --imagenet-root /datasets/imagenet \
      --checkpoint /models/resnet50-0676ba61.pth \
      --device cuda:0 --batch-size 32 --profile-batch-size 4 \
      --observer-images 1024 --profile-images 128 --search-images 512 \
      --seed 20261111 --budget-points 8 \
      --max-candidates "$allowance" --router-mode "$mode" \
      --profile-cache "$RUN_ROOT/shared_fisher.json" \
      --output-dir "$RUN_ROOT/${mode}_${allowance}"
  done
done
```

ILP mode evaluates only seeds regardless of the allowance. Use the v2 ILP results
as the matched baseline: the original checkout has a different default format
space and its refinement lacks v2's seed-relative resource constraints.

## What the search evaluates

Both refinement modes restrict proposals to no more actual estimated storage
and BitOps than their original budget seed, as well as any explicit compute cap.
A winning policy therefore cannot buy accuracy with additional resources.

With W4A8/W8A8, repair groups contain one W8 donor and up to four W4 receivers.
Adjacent-budget seed disagreement prioritizes layers; donor size and receiver
Fisher benefit per byte break ties. Cyclic receiver windows provide different
groups. Uniform two-format endpoints have no such exchanges and retain seeds.

Other format configurations also use cyclic five-layer windows, so activation
changes are searchable even when all enabled weight widths are identical.
The method enumerates up to 32 assignments per group with two formats, or 1,024
with four. Resource checks precede inference. Feasible candidates with more
changed layers are visited first; Fisher scores rank ties but never exclude a
candidate. Small allowances can stop partway through a group.

Budgets take turns evaluating one unique policy. A group's parent stays fixed
until its queue finishes, then groups are rebuilt if the incumbent improved.
The allowance includes seeds and early rejections. Early rejection requires
proof that a candidate cannot add a point to either reported search frontier:
references must cost no more than the candidate on each respective axis, and
the observed mistakes must strictly exceed both thresholds. Ties finish for KL.

The union of both search frontiers and all seeds is saved before held-out
evaluation. Held-out labels never enter proposal generation or policy selection.

## Inspect results and verify

The usual plots, tables, policy files, predictions, and `main.log` remain.
Additional evidence includes:

- `search.json`: algorithm, enabled-format mapping, original seed resource caps,
  stop reason, candidate records, successful edits, and group coverage.
- `groups.json`: parent, layer indices, assignment count, feasible alternatives,
  cache hits at creation, evaluation count, and whether every feasible
  alternative was processed. `complete` describes search coverage, not whether
  every policy required a full evaluation.
- `evaluation_history.jsonl`: candidate parent/group, changed layers, explicit
  edit indices, seed-relative storage/BitOps differences, and race cutoff.
- `frontier.json`: algorithm, group evaluations, unique candidates, policy-image
  evaluations, and profiling/search/proposal/held-out timings.

Mode IDs in search artifacts refer to the saved `quantization_formats` list.
Policy JSONs store explicit bit assignments, so reordering the list does not
change policy meaning. Reload fails if a required format is disabled.

Reload a selected policy with the same data and checkpoint arguments:

```sh
python resnet_benchmark_v2.py evaluate \
  --policy runs/v2_group_64_trial1/policies/P01.json \
  --imagenet-root /datasets/imagenet \
  --checkpoint /models/resnet50-0676ba61.pth \
  --device cuda:0 --batch-size 32 \
  --output-dir runs/v2_reload_trial1
```

Use `export` instead of `evaluate` for export only. The original evaluator can
also reload v2 policy JSONs because the policy schema and quantizers are shared.

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -v
```

For accuracy claims, compare the held-out envelopes at equal actual resource
limits, and report total search time and policy-image work. Charge shared
profiling using `total_seconds_charging_shared_profile`; cache-hit wall time
alone omits profiling. Repeat trials with independently selected training splits
before claiming a consistent gain. Reusing held-out results to tune the method
requires a separate final confirmation set.

Execution remains FP32 fake quantization. BitOps and storage are estimates;
pipeline timings do not establish integer inference latency. Synthetic barrier
tests and small-data smoke runs verify the implementation, not an ImageNet
accuracy advantage or methodological novelty.

## Local verification on 16 November 2026

- Python suite: 67 checks, 66 passed and one CUDA integration check skipped
  because this machine has no CUDA GPU. This includes 19 new v2 tests.
- Real ResNet-50 CPU smoke: four enabled formats, four budgets, 12 candidates;
  and W4A8/W8A8, eight budgets, 32 candidates. Both completed through saved
  policies, held-out predictions, and plots on the small local fixture.
- The two-format run reused the four-format cache. Every repair respected its
  original seed's storage and BitOps, and every seed remained in the shortlist.
- A saved two-format policy reproduced all eight fixture predictions through
  both the v2 CLI and the original CLI. Tiny-network tests also compare logits.
- SHA-256 checks found all 2,684 pre-existing workspace files unchanged,
  excluding Git internals and following no symlinks. Raw smoke artifacts and the
  preservation audit are under `/private/tmp/amp-v2-audit-qwhrdq6d/` on this
  machine; they are local temporary evidence, not a server benchmark.
