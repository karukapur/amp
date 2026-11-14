# ResNet-50 Fisher scoring and feedback router

Search weight-and-activation precision policies for a pretrained ResNet-50 using
Fisher sensitivity scores, MILP seeds, local refinement and cross-budget edit
transfers. Each of 54 operators independently chooses W4A8, W4A16, W8A8, W8A16 or
FP32. Execution uses **FP32 fake quantization**; BitOps and packed storage are
estimates, not measured integer inference speed or resident GPU memory.

## Setup

Use Python 3.10 or newer. Install a matched Torch/Torchvision build for your
machine using the [official PyTorch installer](https://pytorch.org/get-started/locally/),
then install the remaining dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
# Install the Torch/Torchvision build appropriate for your GPU and driver.
python -m pip install -r requirements-runner.txt
python -m unittest discover -s tests -v
```

The CUDA integration check runs automatically when CUDA is available. Optional
walkthrough checks require Node.js: `node tests/test_router_walkthrough.cjs`.
Clone/copy the whole repository to a server, including `amp/`, `tests/`,
`benchmark_data/imagenet_class_index.json` and `third_party/`. Create a new server
environment; do not copy a Mac `.venv` to Linux.

Supply the torchvision ResNet-50 V1 checkpoint `resnet50-0676ba61.pth` using
`--checkpoint`, or place it in `benchmark_data/`. Assets are never downloaded
automatically. ImageNet must use its original class-folder layout:

```text
/datasets/imagenet/
├── train/<WNID>/*.JPEG
└── val/<WNID>/*.JPEG
```

Flat validation archives must first be organized with their official labels.
Subsets retain their original 1,000-class output indices.

## Run

From the repository root:

```sh
python resnet_benchmark.py fisher-frontier \
  --imagenet-root /datasets/imagenet \
  --checkpoint /models/resnet50-0676ba61.pth \
  --device cuda:0 \
  --batch-size 32 \
  --profile-batch-size 4 \
  --observer-images 1024 \
  --profile-images 128 \
  --search-images 512 \
  --router-mode transfer \
  --budget-points 8 \
  --max-candidates 64 \
  --output-dir runs/imagenet_router
```

`python -m amp fisher-frontier ...` is an equivalent entry point. Existing
`python resnet_benchmark.py ...` commands and saved layer/stage policies remain
supported.

| Parameter | Purpose | Default |
|---|---|---|
| `--device` | `auto`, `cpu`, `mps`, `cuda` or a logical GPU such as `cuda:0` | `auto` |
| `--batch-size` | Images per calibration, search and validation batch | 8 |
| `--profile-batch-size` | Images per Fisher backward microbatch | 4 |
| `--observer-images` | Sampled training images for fixed activation ranges | 1024 |
| `--profile-images` | Separate training images for Fisher scoring | 128 |
| `--search-images` | Separate training images for candidate selection | 512 |
| `--router-mode` | `ilp`: seeds only; `local`: local edits; `transfer`: local edits plus transfers | `transfer` |
| `--budget-points` | Storage ceilings between estimated all-W4A8 and all-W8A8 size | 8 |
| `--max-candidates` | Unique search evaluations, including seeds and early rejections | 64 |
| `--profile-cache` | Reusable calibration, Fisher scores and costs | `<output-dir>/fisher_profile.json` |
| `--compute-fraction` | Optional BitOps cap relative to all-W8A8 | **No cap** for `fisher-frontier` |
| `--seed` | Reproducible training-image sampling | 20261111 |

All validation images are streamed in batches after selection is frozen. With a
complete ImageNet installation, that means 50,000 images per selected policy plus
FP32. The search allowance does not limit final validation to 64 images or batches.
Custom split manifests are supported through `--manifest` and `--image-root`;
see [manifest format and legacy commands](docs/legacy-runner.md).

Explicit CUDA requests fail instead of falling back to CPU. A forward/backward
preflight records the GPU/runtime in `runtime.json`. For profiling OOM, reduce
`--profile-batch-size` to 1; for other phases reduce `--batch-size`. FP32 reference
arithmetic disables TF32; unset `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` if present.
There is no distributed/multi-GPU execution. Use fresh profile caches when changing
fingerprinted profiling settings or devices; incompatible caches fail explicitly.

## Outputs and logging

| Output in `--output-dir` | Contents |
|---|---|
| `main.log` | Live console messages with UTC timestamps, configuration, phase progress/ETA, solver/router decisions, warnings and failure tracebacks |
| `frontier.png`, `frontier.svg` | Storage compression ratio on X, held-out top-1 accuracy on Y |
| `frontier_bitops.png`, `frontier_bitops.svg` | Actual estimated GigaBitOps on X, held-out top-1 accuracy on Y |
| `frontier.csv`, `frontier.json` | Measurements, storage/BitOps frontier flags, resource settings and timings |
| `sensitivity_costs.csv`, `fisher_profile.json` | Per-operator scores, costs and frozen calibration |
| `solver.json`, `evaluation_history.jsonl`, `search.json`, `transfers.json` | Allocation results, search measurements and edit provenance |
| `selection.json`, `policies/` | Frozen shortlist and reloadable precision assignments/scales |
| `frontier_predictions.csv` | Held-out labels and predictions for each evaluated model |

Progress appears automatically in the terminal and is flushed to `main.log`.
Monitor it from another terminal with:

```sh
tail -f runs/imagenet_router/main.log
```

Logs append if a directory is reused; other outputs can be replaced. Use a new
output directory per experiment. Interrupted runs may leave partial prediction
CSVs; failures and Ctrl+C are recorded in the log.

Storage compression is FP32 parameter bytes divided by estimated quantized
parameter bytes. BitOps is the sum of `MACs × weight bits × activation bits`.
Each graph marks its own nondominated measured points. The union of both search
frontiers and the MILP seeds is frozen before held-out evaluation; the resulting
plots describe that measured shortlist, not a global optimum. Search still sweeps
storage budgets and has no implicit compute ceiling.

Reload or export a selected policy with the same checkpoint and dataset arguments:

```sh
python resnet_benchmark.py evaluate \
  --policy runs/imagenet_router/policies/P01.json \
  --manifest runs/imagenet_router/dataset_manifest.json \
  --checkpoint /models/resnet50-0676ba61.pth \
  --device cuda:0 --batch-size 32 --output-dir runs/recheck
```

Use `export` instead of `evaluate` to export the policy without held-out inference.

## Repository layout

```text
amp/                 Implementation and python -m amp entry point
tests/               Python tests and walkthrough checks
scripts/             Historical audit and analysis tools
experiments/         Independent transport-sketch prototype
docs/                Method, legacy usage, visual walkthroughs and research notes
reports/             Selected historical evidence and research limitations
third_party/         NVIDIA attribution and license
benchmark_data/      Canonical class map; local checkpoints/images are ignored
resnet_benchmark.py  Compatibility CLI/import
fisher_router.py     Compatibility import
runs/                New outputs (ignored)
artifacts/           Archived raw development/audit runs (local, ignored)
resnet_results/      Retained legacy pilot inputs/results (local, ignored)
```

The implementation follows [NVIDIA's licensed scoring functional](third_party/modelopt/NOTICE.md).
The router's novelty and benefit remain unproven. The [Fisher pilot](reports/fisher-pilot/REPORT.md)
found no held-out frontier advantage over local refinement. A historical stage
result also required a [device-rounding correction](reports/pareto-audit/REPORT.md);
it does not establish an accuracy improvement over FP32. CUDA and full-ImageNet
execution still require validation on the target server.

Further reading: [method details](docs/method.md), [visual router story](docs/router-walkthrough.html),
[recorded pilot replay](docs/router-replay.html), [legacy workflows](docs/legacy-runner.md),
[audit tools](scripts/README.md), and [archived-results guide](reports/README.md).
