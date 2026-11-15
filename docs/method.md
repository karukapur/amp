# Fisher scoring and feedback search

Open [the illustrated router story](router-walkthrough.html) in a browser.
Scroll through one precision edit: discovery, compatibility checks, transfer,
measurement and the final frontier. It uses a clearly labelled six-operator
example with three optional demonstrations and works offline. The detailed
[recorded pilot replay](router-replay.html) is also available.
Check both pages' logic and controls with `node tests/test_router_walkthrough.cjs`.

The `fisher-frontier` command independently searches all **54 operators**, including
the stem, projection convolutions and classifier. Each chooses W4A8, W4A16, W8A8,
or W8A16. FP32 is only the teacher/baseline, outside the searchable formats.

```sh
.venv/bin/python resnet_benchmark.py fisher-frontier \
  --router-mode transfer --budget-points 8 --max-candidates 64 \
  --output-dir runs/fisher_transfer
```

For full ImageNet, add `--imagenet-root /datasets/imagenet --device cuda
--batch-size 32 --profile-batch-size 4`. Dataset/checkpoint/device arguments are shared with the existing
runner. A16 means signed **integer** quantization, not FP16. Each operator has its
own input quantizer; residual identity paths and additions remain floating point.
This graph differs from the older stage runner's shared block-input quantizer, so
compare router variants using this new graph rather than historical stage results.

The scorer adapts NVIDIA AutoQuantize's elementwise squared product of output
gradient and joint W/A output error. It uses labeled summed cross-entropy, one
backward per profiling microbatch, local operator replays, and image-count
normalization. It neither constructs a full Hessian nor trains model weights.
Ranges are calibrated once on the observer split and frozen. No Shapley scoring,
per-policy recalibration, or forced monotone ordering of candidate scores is used.
See [upstream attribution](../third_party/modelopt/NOTICE.md).

The initial ILPs minimize summed Fisher scores under eight default storage budgets
between all-W4A8 and all-W8A8 size. **There is no default BitOps ceiling.** BitOps
is reported for every policy but does not restrict seeds, local edits or transfers.
An explicitly supplied `--compute-fraction` opts into a cap relative to all-W8A8;
the minimum feasible fraction is 0.5 when using this optional constraint. Legacy
stage-based `search` retains its existing default. FP32 reference cost is computed
separately at 32 bits and is not an allocation choice. Stored-size estimates include packed weights, floating biases,
weight scales and activation scales, excluding file/container overhead. BitOps
counts convolution/linear MACs; it excludes floating residual and quantization work.

The feedback router explores one-layer changes and two-layer exchanges. Successful
edits can transfer to another budget only when the changed layers have matching
starting formats and the resulting policy fits the destination storage budget
(and an explicit compute cap, if supplied). Transfers
are prioritized by similarity of the surrounding assignments. Every fourth proposal
is reserved for local exploration. Measurements update all feasible budget
incumbents, while original Fisher scores stay unchanged. The evaluation allowance
counts unique policies, including seeds and early rejections; it must cover all
unique seeds. An unfinished policy is rejected only after its observed mistakes
exceed every eligible incumbent's total mistakes; ties finish for the KL tiebreak.

Three modes are available:

- `--router-mode ilp`: evaluate the unique seed allocations only.
- `--router-mode local`: refine independently without transferring edits.
- `--router-mode transfer`: enable the feedback router (default).

For a matched comparison, reuse one profile cache, dataset, optional compute cap, budget
count, batch size and candidate allowance across `local` and `transfer` runs:

```sh
.venv/bin/python resnet_benchmark.py fisher-frontier \
  --router-mode local --max-candidates 64 \
  --profile-cache runs/shared_fisher.json --output-dir runs/fisher_local
.venv/bin/python resnet_benchmark.py fisher-frontier \
  --router-mode transfer --max-candidates 64 \
  --profile-cache runs/shared_fisher.json --output-dir runs/fisher_transfer
```

Caches fingerprint the checkpoint, observer/profile image contents and labels,
operator graph, preprocessing, device, Torch version, profiling batch size and
quantization settings. CUDA caches also record GPU capability and CUDA/cuDNN versions.
Mismatches fail explicitly. Each run reports actual wall time;
`total_seconds_charging_shared_profile` additionally charges the original
calibration/profiling time to cache-hit runs for fair comparisons. Local and
transfer modes share global result caching and incumbent updates: their only
proposal-mechanism difference is edit transfer.

Outputs include `sensitivity_costs.csv`, `fisher_profile.json` (unless another cache
path is supplied), `solver.json`, `evaluation_history.jsonl`, `search.json`,
`transfers.json`, `selection.json`, policy CSV/JSON files, and frontier
CSV/JSON/PNG/SVG plus held-out predictions. The shortlist is saved before held-out
inference; search accuracy is primary and teacher KL breaks ties. Selected search
frontier policies for **both storage and BitOps**, plus all ILP seeds, receive
held-out evaluation. Their union is frozen before reading held-out labels; a
policy shared by both frontiers is evaluated once. Allocation still sweeps
storage budgets. Adding the BitOps plot does not switch the search objective.
Both graphs use the same final measurements and policy IDs:

- `frontier.png` / `frontier.svg`: storage compression ratio versus top-1 accuracy.
  Larger compression ratio means less estimated parameter storage.
- `frontier_bitops.png` / `frontier_bitops.svg`: estimated BitOps versus top-1
  accuracy, displayed in **GigaBitOps (10^9 BitOps)**. Lower means less estimated
  compute. Each policy's cost is the sum of `MACs × weight bits × activation bits`
  over operators, rather than the budget ceiling. This is not measured latency.

Each graph marks its own nondominated evaluated points. `frontier.csv` and
`frontier.json` contain both `validation_frontier` (storage) and
`validation_bitops_frontier` flags, plus corresponding search-frontier flags.
The plotted frontiers cover the evaluated shortlist; they are not global optima.
The additional compute-frontier candidates can increase final validation work,
without increasing the unique search-candidate allowance. FP32 is displayed only
as a baseline, with its numeric accuracy; it never joins either quantized frontier.
`accuracy_drop_pp` reports the accuracy difference from that baseline.
`frontier.json` records `compute_fraction` and `bitops_budget` as `null` when
uncapped. `budget_metrics` names the columns in the saved budget arrays: storage
alone by default, or storage and BitOps when capped. Policy costs always retain
both values. Reload layer policies with the existing `evaluate` or `export`
commands, passing `--policy runs/fisher_transfer/policies/P01.json`.

Every CLI run automatically writes **`<output-dir>/main.log`** while displaying
the same application messages in the console. Entries include UTC timestamps and
levels, the command/configuration, device checks, dataset preparation, calibration
and Fisher progress, MILP status, candidate evaluations, successful edits,
incumbent updates, held-out results, timings and output paths. Long evaluations
report image counts and approximate remaining time. Python warnings and failure
tracebacks are saved too, including failures during checkpoint/device validation.
The log is flushed as messages arrive and appended if an output directory is
reused. Ctrl+C records an interruption and exits with status 130. Follow it from
another terminal with:

```sh
tail -f runs/imagenet_fisher/main.log
```

This is a bounded research heuristic, not an exhaustive layerwise frontier or an
established novel method. It may fail to improve on ILP or local search. All
execution remains fake quantization in FP32; reported timings measure search and
evaluation, not integer inference speed. Validate on full ImageNet before making
accuracy claims.

The [retained-pilot validation](../reports/fisher-pilot/REPORT.md) records 22 passing
checks and a matched 24-candidate comparison. The router exercised two transfers
but selected the same final policies as local refinement; no frontier improvement
or evaluation-work reduction was demonstrated. That historical pilot used the
former 75% compute cap; its results are not an uncapped baseline. The default output directory for
the new command is `runs/fisher_frontier`.

```sh
.venv/bin/python -m unittest discover -s tests -v
```

