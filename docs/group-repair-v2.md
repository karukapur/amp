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

`ilp_v2` is a result label identifying the v2 pipeline, not a new ILP algorithm.
The seed allocation retains the same additive Fisher objective, binary format
selection, storage/optional BitOps constraints, and SciPy MILP solver settings
(10-second limit, zero requested relative gap). V2 makes the enabled formats
editable and derives distinct storage ceilings from them. Changing those inputs
can change the resulting seeds; the optimization formulation itself is unchanged.

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
as the matched baseline, with the same enabled formats, cache, and budget
settings as the refinement runs. Both implementations ship all four formats
enabled, but commenting out entries changes the problem supplied to the solver.
The original refinement also lacks v2's seed-relative resource constraints.

## Pipeline and search state

The execution sequence is:

```text
Calibrate activation bounds
  -> Profile Fisher sensitivities
  -> Solve one MILP per distinct storage ceiling
  -> Evaluate each unique MILP seed on search images
  -> Refine policies with local search or group repair
  -> Freeze the union of search frontiers and all seeds
  -> Evaluate the frozen shortlist on held-out images
```

Solving a MILP produces a precision assignment. Evaluating that assignment runs
the complete quantized network to measure its performance. These are separate
steps; all seed solves finish before seed evaluation and refinement begin.

The requested budget points are equally spaced **storage ceilings**, not exact
policy sizes. A seed may use less storage than its ceiling. Different ceilings
can produce identical seeds, and some seeds can be dominated rather than lie on
the measured Pareto frontier. Identical policies are evaluated once.

The router is best described as a **stateful search controller**:

| Information | Behavior during search |
|---|---|
| Calibration outputs: activation quantization bounds | Fixed; no per-policy recalibration. These are bounds, not a cache of every activation tensor. |
| Sensitivity outputs: per-operator, per-format Fisher scores | Fixed; measurements do not update the sensitivities. |
| Policy archive: assignments, measured mistakes/KL, costs, and completion status | Updated after each unique evaluation; prevents duplicate evaluations. |
| Budget incumbents: best measured feasible policies | Updated when a completed candidate improves accuracy or its KL tiebreaker. |
| Group state: fixed parent, selected layers, pending assignments, and coverage | Advances as proposals are evaluated; rebuilt around an improved incumbent after the current group finishes. |

Calling this a state machine is reasonable informally. The implementation uses
search loops and stored state rather than an explicit state-transition model.
The evolving state belongs to policy search, while calibration and sensitivity
results remain fixed inputs. The controller operates offline; the saved policy
does not introduce an inference-time routing decision.

## What MILP already considers, and why repair can differ

The MILP formulation already permits **every enabled per-layer assignment that
satisfies its resource constraints**. It does not need to enumerate each one
individually to optimize over that space. When the solver reports optimality,
the selected policy is optimal for the supplied objective. A timed-out feasible
incumbent has no such certificate; the run records this distinction.

With binary variable `x[i,f]` selecting format `f` for operator `i`, the seed
problem is:

```text
minimize  sum_i sum_f Fisher[i,f] * x[i,f]
subject to:
    sum_f x[i,f] = 1                         for every operator i
    sum_i sum_f storage[i,f] * x[i,f] <= storage ceiling
    sum_i sum_f BitOps[i,f] * x[i,f] <= compute cap   if supplied
    x[i,f] in {0, 1}
```

The limitation is the **additive quality approximation**, not the solver's
ability to represent combinations. The MILP accounts for how assignments share
the resource budget, but its objective adds fixed sensitivities without
cross-layer interaction terms. It does not measure the accuracy of every
complete configuration.

| | MILP seed allocation | V2 group repair |
|---|---|---|
| Assignment space | Entire feasible space in the supplied formulation | Selected neighborhoods around measured incumbents |
| Quality criterion | Summed fixed Fisher sensitivities | Measured search-set accuracy, with teacher KL breaking ties |
| Combined effect on predictions | Approximated by additive scores | Observed by running the proposed complete policy |

For illustration only, two policies within the same resource limits could have:

| Policy | Summed Fisher score, lower is preferred | Measured accuracy, higher is preferred |
|---|---:|---:|
| MILP solution | 0.10 | 74% |
| Another configuration | 0.12 | 75% |

These numbers are hypothetical, not experimental results. The MILP correctly
chooses the first policy for its objective. Repair could discover the second
by measuring it. It does not expand the assignment space beyond what the MILP
could represent; it tests whether actual performance contradicts the Fisher
ranking. A richer MILP could also model interactions if suitable terms were
supplied, but the current seed objective contains only additive sensitivities.

Grouping alone therefore guarantees no advantage. It is useful only if the
additional evaluations find meaningful Fisher-ranking errors at acceptable
search cost, and the selected improvements generalize to held-out data.

## What the search evaluates

Both refinement modes restrict proposals to no more actual estimated storage
and BitOps than their original budget seed, as well as any explicit compute cap.
A winning policy therefore cannot buy accuracy with additional resources.

### Donors, receivers, and group construction

For W4A8/W8A8, the code does not classify layers using a fixed "large" or "small"
threshold. It identifies potential donors and receivers in the current parent:

| Role | Eligible layers | Ranking, in order |
|---|---|---|
| Donor | Currently W8 | Neighboring-budget seed disagreement descending; storage released by W8 -> W4 descending; operator index ascending. |
| Receiver | Currently W4 | Neighboring-budget seed disagreement descending; predicted Fisher improvement per additional byte for W4 -> W8 descending; operator index ascending. |

Disagreement is the number of immediately neighboring-budget seeds whose
assignment for that operator differs from the current parent. It ranges from
zero to two; an endpoint has at most one neighboring seed. The comparison uses
the original neighboring seeds, not their evolving incumbents. Disagreement
ranks ahead of storage saving or benefit per byte, so the largest donor need
not be tried first.

For a layer with `P` weights, W8 -> W4 releases approximately `P / 2` bytes;
the actual cost table accounts for byte rounding. Biases and scale metadata
remain unchanged. Receiver benefit per byte is:

```text
(Fisher[W4A8] - Fisher[W8A8]) / (storage[W8A8] - storage[W4A8])
```

This is a fixed sensitivity prediction, not a benefit learned from candidate
accuracy. Receivers are not necessarily the smallest layers. Selecting a donor
and receivers does not guarantee that their exchange fits either resource cap.

For each ranked donor, construct cyclic windows of up to four ranked receivers:

```text
D1 + R1, R2, R3, R4
D1 + R2, R3, R4, R5
D1 + R3, R4, R5, R6
... wrap around the receiver list, then repeat for D2 ...
```

Identical groups are deduplicated. Groups can span different network stages;
their operators need not be adjacent or part of the same residual block.
Uniform two-format parents have no donor/receiver exchanges, so those budget
slots produce no group proposals until their incumbent changes. All original
seeds remain in the final shortlist.

### Assignments inside each group

Within a group, every enabled-format assignment is considered while all other
operators remain fixed. The donor/receiver labels guide group selection; they
do not force every candidate to downgrade the donor or upgrade every receiver.
The unchanged parent and resource-infeasible alternatives are removed before
evaluation, and globally cached policies are skipped.

A possible coordinated change is:

| Policy | One large layer | Three other layers |
|---|---|---|
| Parent | W8 | W4, W4, W4 |
| Proposal | W4 | W8, W8, W8 |

The released storage could fund the three upgrades. The complete proposal must
also pass the BitOps check. Such a four-layer change can be measured directly
even if its intermediate one-/two-layer moves worsen performance. It remains
an assignment the MILP could represent, with a different quality assessment.

Other format configurations also use cyclic five-layer windows, so activation
changes are searchable even when all enabled weight widths are identical.
Those windows rank layers by neighboring-seed disagreement, then Fisher-score
range, then operator index, and are appended after the donor/receiver groups.
The method enumerates up to 32 assignments per group with two formats, or 1,024
with four. Resource checks precede inference. Feasible candidates with more
changed layers are visited first; Fisher scores rank ties but never exclude a
candidate. The policy tuple breaks remaining ties. Small allowances can stop
partway through a group or before later groups are reached.

### Budget scheduling, updates, and new frontier points

Each budget has its own search state, but evaluations are interleaved rather
than completing an entire budget before starting the next:

```text
Budget 1: one new candidate
Budget 2: one new candidate
...
Return to Budget 1 for its next candidate
```

Slots without an untested feasible proposal are skipped. A group's parent stays
fixed until its queue finishes, even if a better incumbent is found meanwhile.
Then groups are rebuilt if the incumbent changed, or the next group is tried.
An incumbent can come from another budget's seed or search, because completed
measurements update every slot whose original seed resource limits they satisfy.
Those limits remain fixed even when the incumbent changes.

The archive and candidate allowance are shared across budgets. For example,
`--max-candidates 64` with eight unique seeds leaves at most 56 new evaluations
for the entire run, not 64 for each budget. Early rejections also consume one
unique evaluation. Duplicate policies consume no additional evaluation.

Early rejection requires proof that a candidate cannot add a point to either
reported search frontier:
references must cost no more than the candidate on each respective axis, and
the observed mistakes must strictly exceed both thresholds. Ties finish for KL.

Repair creates additional policies within the existing limits; it does not
create additional budget ceilings. A candidate can improve an incumbent or
provide an intermediate cost/accuracy trade-off. A lower-accuracy policy can
still be nondominated if it also has sufficiently lower cost. Generating a
candidate alone does not make it a useful Pareto point.

The union of both search frontiers and all seeds is saved before held-out
evaluation. Held-out labels never enter proposal generation or policy selection.
Search-frontier membership does not guarantee held-out-frontier membership.

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
