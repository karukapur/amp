# ResNet-50 mixed-precision policy runner

**Audit correction:** the historical `runs/pareto_pilot` result of 75.2% at 6.94× storage compression depends on PyTorch 2.4.1 MPS using a different half-tie rounding rule from CPU. The runner now explicitly uses ties-to-even. The original policy gives 74.4% on corrected MPS and 74.2% on CPU; repeating the search selects a different same-size policy with 73.4% versus FP32's 73.0%. None demonstrates an accuracy improvement. The 6.94× packed parameter size was verified by writing and reloading real packed bytes, but runtime tensors and computation remain FP32. See the [full independent audit](runs/pareto_audit/REPORT.md) and [corrected results](runs/pareto_audit/corrected_mps_batch8/frontier.csv). Original pilot artifacts are preserved.

Use `fisher-frontier` for independent operator precisions with Fisher scoring and
feedback routing. The older `search` and `frontier` commands select stage-wide
W4A8 / W4A16 / W8A8 / W8A16 choices with the stem and classifier fixed at W8A8;
their detailed CSVs expand stage choices rather than optimizing each layer.

## New: layerwise Fisher scoring and feedback router

The `fisher-frontier` command independently searches all **54 operators**, including
the stem, projection convolutions and classifier. Each chooses W4A8, W4A16, W8A8,
W8A16 or FP32. The older commands below keep their original stage-wide behavior.

```sh
.venv/bin/python resnet_benchmark.py fisher-frontier \
  --router-mode transfer --budget-points 8 --max-candidates 64 \
  --compute-fraction 0.75 --output-dir runs/fisher_transfer
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
See [upstream attribution](third_party/modelopt/NOTICE.md).

The initial ILPs minimize summed Fisher scores under eight default storage budgets
between all-W4A8 and all-W8A8 size, and a fixed BitOps cap (default 75% of all-W8A8).
The minimum feasible compute fraction is 0.5. FP32 remains an operator choice and
costs 32 bits. Stored-size estimates include packed weights, floating biases,
weight scales and activation scales, excluding file/container overhead. BitOps
counts convolution/linear MACs; it excludes floating residual and quantization work.

The feedback router explores one-layer changes and two-layer exchanges. Successful
edits can transfer to another budget only when the changed layers have matching
starting formats and the resulting policy satisfies both constraints. Transfers
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

For a matched comparison, reuse one profile cache, dataset, compute cap, budget
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
frontier policies and all ILP seeds receive held-out evaluation. The FP32 reference
is plotted separately and is excluded from the constrained frontier when it violates
the compute cap. Reload layer policies with the existing `evaluate` or `export`
commands, passing `--policy runs/fisher_transfer/policies/P01.json`.

This is a bounded research heuristic, not an exhaustive layerwise frontier or an
established novel method. It may fail to improve on ILP or local search. All
execution remains fake quantization in FP32; reported timings measure search and
evaluation, not integer inference speed. Validate on full ImageNet before making
accuracy claims.

The [retained-pilot validation](runs/fisher_pilot/REPORT.md) records 22 passing
checks and a matched 24-candidate comparison. The router exercised two transfers
but selected the same final policies as local refinement; no frontier improvement
or evaluation-work reduction was demonstrated. The default output directory for
the new command is `runs/fisher_frontier`.

```sh
.venv/bin/python -m unittest test_resnet_runner test_fisher_router -v
```

### Remote CUDA server and full ImageNet

Copy `resnet_benchmark.py`, `fisher_router.py`, `requirements-runner.txt`,
`third_party/modelopt/`, and the three test files (`test_resnet_runner.py`,
`test_fisher_router.py`, `test_remote_readiness.py`). Also copy the V1 checkpoint
and `benchmark_data/imagenet_class_index.json`; the label mapping is required even
when `--checkpoint` points elsewhere. The runner refuses partial class mappings
instead of renumbering an ImageNet subset.

Install a matched CUDA-enabled Torch/Torchvision pair using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then
`python -m pip install -r requirements-runner.txt`. Do not copy the Mac `.venv`
to Linux. Run the checks on the server:

```sh
python -m unittest test_resnet_runner test_fisher_router test_remote_readiness -v
```

The CUDA integration test is skipped on CPU-only machines and automatically runs
on a GPU host. It compares CPU/CUDA Fisher scores on a small convolutional model,
checks costs, and reloads a CPU-exported policy on CUDA. The CPU checks include a
synthetic 1,000-class folder layout and streaming all 50,000 validation rows with
a partial final batch. These are code-path checks, not a full-ImageNet accuracy
or GPU-memory certification.

Use the organized `train/<WNID>/*.JPEG` and `val/<WNID>/*.JPEG` layout described
below. Flat validation archives must first be organized with their official labels.
The following command samples training images for calibration/profiling/search
and evaluates every validation image:

```sh
python resnet_benchmark.py fisher-frontier \
  --imagenet-root /datasets/imagenet \
  --checkpoint /models/resnet50-0676ba61.pth \
  --device cuda:0 --batch-size 32 --profile-batch-size 4 \
  --observer-images 1024 --profile-images 128 --search-images 512 \
  --router-mode transfer --budget-points 8 --max-candidates 64 \
  --compute-fraction 0.75 --output-dir runs/imagenet_fisher
```

Explicit `--device cuda:N` fails if that logical GPU is unavailable; it never
falls back to CPU. `CUDA_VISIBLE_DEVICES` determines the logical numbering.
Before scanning ImageNet, the CLI checks CUDA convolution, quantization, matrix
multiplication and backward execution. `runtime.json` records the selected GPU,
capability, Torch CUDA build and cuDNN version. Kernel/driver incompatibilities
therefore fail before expensive profiling. FP32 precision controls support both
legacy and newer PyTorch APIs, following the
[PyTorch CUDA precision documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html).
Unset `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` if your server sets it.

If backward profiling exhausts GPU memory, lower `--profile-batch-size` to 1.
For calibration, search or validation OOMs, lower `--batch-size`. Profiling hooks
release their saved outputs after each microbatch. Use a fresh cache path when
changing profiling settings or moving between CPU and CUDA. An incompatible
cache fails explicitly; portable policy JSON files remain usable across devices.

Validation images are decoded and evaluated batch by batch, including the final
partial batch; all 50,000 images are used for a complete ImageNet validation set.
Only the sampled search images are retained together in host RAM. Corrupt images
and non-finite logits fail with context instead of silently dropping samples.
`frontier.json` reports actual image counts and class counts per split. Use a new
output directory for each run; interrupted runs can leave partial prediction files.

## Existing stage-based Pareto frontier

You do not need to choose memory or compute fractions. After the installation and dataset setup below, run:

```sh
python resnet_benchmark.py frontier \
  --imagenet-root /datasets/imagenet \
  --device cuda --batch-size 32 \
  --output-dir runs/imagenet_frontier
```

For the retained small pilot on this Mac:

```sh
.venv/bin/python resnet_benchmark.py frontier --output-dir runs/pareto_pilot
```

The X-axis is **compression ratio = estimated FP32 model storage / estimated quantized model storage**. For example, 4× means one-quarter of the original model storage, a 75% reduction. The Y-axis is held-out top-1 accuracy, in percent. Larger compression ratios and higher accuracy are better. A point is dominated if another evaluated point is at least as good on both axes and strictly better on one; the plot connects the remaining points. Intermediate points on a connecting line do not represent evaluated models.

This workflow evaluates all 256 stage policies on the separate search split, reusing shared computation. At each distinct storage size, it keeps the most accurate search policy (teacher KL breaks ties), then retains the nondominated size/accuracy trade-offs. It evaluates those candidates, plus all-stage W4A8/W8A8 anchors and the FP32 reference, on **all validation images**. No test labels are used to pick the candidates. This costs more search time than selecting a single budgeted policy, but avoids evaluating all 256 policies on the full validation set. No Fisher/sensitivity profiling is needed in frontier mode.

Outputs:

- `frontier.png` and `frontier.svg`: the plot, with IDs identifying each measured policy.
- `frontier.csv`: compression, top-1 accuracy, stage precisions, estimated size, and BitOps for each point. `validation_frontier` marks the nondominated evaluated points.
- `policies/P01.csv`, `policies/P01.json`, etc.: detailed per-layer assignments and reloadable calibrated policies. The FP32 reference is not quantized and has no policy file.
- `frontier_predictions.csv`: all held-out image labels and predictions for every plotted model.
- `frontier_search.npz`: all 256 policies' search-set results; `frontier.json`: run scope, counts and timings.

Choose a point by its acceptable accuracy/size trade-off, then use its `policies/Pxx.json` with the existing `evaluate` or `export` command. This is exploratory model selection: if you choose/tune based on validation results, use a fresh untouched evaluation set for a final confirmatory accuracy claim.

The frontier is exact for the **search sample and four-stage search space**; the held-out frontier is only among shortlisted policies, not a guarantee of the best possible full-ImageNet or independently assigned per-layer frontier. Activation working memory and latency are not represented by the compression axis. W4A8 and W4A16 have the same model storage here, but different activation costs; the CSV therefore preserves stage modes and estimated BitOps alongside accuracy. All sizes include folded weights/biases, plus scales for quantized models, and exclude serialization overhead. The original FP32 checkpoint's file size is not the denominator.

## Run on your GPU with ImageNet

Copy this project to the GPU machine, including:

- `resnet_benchmark.py`
- `fisher_router.py` (required for `fisher-frontier` and layer-policy evaluation)
- `requirements-runner.txt` and `third_party/modelopt/`
- `benchmark_data/resnet50-0676ba61.pth` (the existing official V1 checkpoint, approximately 98 MiB)
- `benchmark_data/imagenet_class_index.json` (the original 1,000-class label mapping)

There is no need to copy the pilot images when using your own ImageNet installation. No automatic dataset or checkpoint downloads occur.

Create a Python environment, then install **a matched CUDA-enabled Torch/Torchvision pair** using the command for your GPU/driver from the [official PyTorch installer](https://pytorch.org/get-started/locally/). Do not copy the Mac `.venv` to Linux. The local implementation was verified with Torch 2.4.1 / Torchvision 0.19.1; a newer matched build may be required by newer GPUs.

```sh
python3 -m venv .venv
source .venv/bin/activate
# Install torch and torchvision using the official CUDA installation command.
python -m pip install -r requirements-runner.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect your ImageNet folder to look like:

```text
/datasets/imagenet/
  train/
    n01440764/*.JPEG
    n01443537/*.JPEG
    ...
  val/
    n01440764/*.JPEG
    n01443537/*.JPEG
    ...
```

The folder names are ImageNet WordNet IDs, not arbitrary class numbers. A flat, unorganized official validation archive must first be organized using its official labels. Subsets work too: the runner maps each WNID to its **original 1,000-class output index**, rather than renumbering the folders present.

Run selection and then evaluation on **every image in `val/`**:

```sh
python resnet_benchmark.py search \
  --imagenet-root /datasets/imagenet \
  --device cuda \
  --batch-size 32 \
  --observer-images 1024 \
  --profile-images 128 \
  --search-images 512 \
  --memory-fraction 0.85 \
  --compute-fraction 0.75 \
  --evaluate \
  --output-dir runs/imagenet_85_75
```

Only the sampled training images are used for activation calibration (1,024), sensitivity profiling (128), and policy selection (512). Those samples are disjoint and selected reproducibly with seed 20261111, spreading sampling across classes. All validation images are held out. On a standard complete installation evaluation processes 50,000 validation images; on a subset it reports the actual count. Validation images and predictions are processed/written batch by batch, so the full validation set is never held in memory. Profile/search images are still loaded into host memory; keep those counts modest.

`--device auto` prefers CUDA, then Apple MPS, then CPU. `--device cuda:1` selects a specific GPU. Explicit CUDA requests fail if CUDA is unavailable, instead of silently running on CPU. Reduce `--batch-size` if GPU memory is tight. The new `fisher-frontier` supports an independent `--profile-batch-size` (default four); legacy stage profilers retain their original batch size of four. No distributed/multi-GPU execution is implemented.

Budget fractions are relative to the estimated all-W8A8 packed model size and BitOps. The defaults mean **at most** 85% and 75%, respectively; the selected model may use less. BitOps are a compute proxy, not measured integer inference latency. To try another budget, change the two fractions and use another output directory.

## Files produced automatically

| File | Contents |
|---|---|
| `selected_policy.csv` | Weight and activation precisions for each operation, quantization ranges/granularity, calibrated activation scales, and floating-point operations |
| `selected_policy.json` | Stage policy, resource estimates, search certificate, checkpoint hash, device/version, and saved activation calibration bounds |
| `dataset_manifest.json` | Exact sampled training and validation image paths/labels when using `--imagenet-root` |
| `evaluation.json` | Float/mixed top-1 and top-5 accuracy, correct counts, image count, KL and timing; created with `--evaluate` |
| `evaluation_predictions.csv` | One row per validation image: label, float/mixed predictions and correctness; created with `--evaluate` |

For an existing output directory, filenames are replaced by the new run; use a distinct directory to retain each experiment. An interrupted evaluation can leave a partial predictions CSV; the complete `evaluation.json` is written only on completion. Start a new output directory after an interrupted run to avoid confusing older summaries with partial output.

The policy CSV contains 54 weight-layer rows and 50 activation-quantizer rows. It also explicitly lists folded batch normalization, ReLUs, pooling, residual addition and final logits. For weight rows, `weight_bits` is the storage precision and `activation_bits` describes the named `input_quantizer`. Activation quantizer rows give their integer range and scalar scale. Weight scales are per output channel and are regenerated from the checkpoint rather than expanded into thousands of CSV entries. All zero points are zero.

ReLU outputs, residual additions, biases, accumulators and final logits use FP32 in this implementation. Quantized tensors are simulated by quantization/dequantization in floating point. **This runner searches and evaluates INT4/INT8/INT16 policies; it does not export integer CUDA kernels or enable FP16 autocast.** CUDA accelerates the simulation. TF32 is disabled to preserve the FP32 reference arithmetic. Accuracy/timings can still differ across devices, library versions and batch sizes; compare float and mixed models on the same machine and images.

## Evaluate or export a saved policy

For the same dataset, reuse the generated manifest. Saved calibration bounds are restored, and a different checkpoint hash is rejected:

```sh
python resnet_benchmark.py evaluate \
  --policy runs/imagenet_85_75/selected_policy.json \
  --manifest runs/imagenet_85_75/dataset_manifest.json \
  --device cuda --batch-size 32 \
  --output-dir runs/imagenet_85_75_recheck
```

Use `export` instead of `evaluate` to write just the policy CSV. Policies saved by the earlier pilot have no stored calibration bounds; those are recalibrated from the observer images and a message is printed. New search runs save bounds automatically. The original benchmark phases `run` and `compare` also emit policy CSVs in their `policies/` output subdirectory.

## Custom manifests and small local runs

Instead of `--imagenet-root`, provide `--manifest path/to/manifest.json`. Each split is a list of records:

```json
{
  "observer": [{"file": "train/image_a.JPEG", "label": 0}],
  "profile": [{"file": "train/image_b.JPEG", "label": 217}],
  "search": [{"file": "train/image_c.JPEG", "label": 0}],
  "test": [{"file": "val/image_d.JPEG", "label": 217}]
}
```

Use original ImageNet class indices (0–999). Relative paths resolve from the custom manifest's directory, or from an explicit `--image-root`. Absolute paths also work. Paths must be distinct across splits; this path check does not detect differently named copies of identical images, so preserve the original dataset split. Observer/profile/search splits must be nonempty; test is required only for evaluation. No label-free search is implemented.

With the retained pilot files on this Mac:

```sh
.venv/bin/python resnet_benchmark.py search --evaluate --output-dir runs/local
.venv/bin/python -m unittest test_resnet_runner -v
```

The original pilot's [CSV](resnet_results/selected_policy.csv) is already generated. A full local MPS selection plus streamed 500-image evaluation passed after these changes, reproducing 369/500 mixed and 365/500 float correct predictions. Reloading the saved policy and changing the batch size from 8 to 16 reproduced those counts. Eight CPU unit checks cover all quantizer exports, original class mapping, split isolation, batching, saved-scale restoration/checkpoint matching, mocked CUDA routing and Pareto dominance/tie handling. **NVIDIA execution has not been tested here because this machine has no CUDA GPU.** See [the research report](resnet_results/REPORT.md) for the pilot's results and limits.
