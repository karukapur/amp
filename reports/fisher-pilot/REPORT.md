# Fisher router implementation: retained-pilot validation

13 November 2026.

Historical capped experiment. Full raw artifacts are now stored locally under `artifacts/runs/fisher_pilot/` (ignored by Git). The commands below explicitly retain the old cap and use a fresh cache; later code versions also evaluate the BitOps shortlist.

**The implementation is working, but this pilot does not demonstrate a router advantage.** Both modes selected the same eight policies and produced identical predictions for all 500 held-out images. Both nondominated points were already present among the ILP seeds; neither refinement improved this measured frontier.

## Experiment

Pretrained torchvision ResNet-50 V1, CPU, two Torch threads, inference batch size 8. The retained ten-class pilot supplies 100 observer images, 50 labeled profiling images, 100 search images and 500 held-out images; classifier outputs retain their original 1,000-class indices. Each mode uses eight storage budgets, a 0.75 all-W8A8 BitOps cap and 24 unique candidate evaluations. This bounded validation uses fewer candidates than the CLI default of 64.

Run local refinement first, then transfers, using the same cache:

```sh
.venv/bin/python resnet_benchmark.py fisher-frontier --device cpu --batch-size 8 --budget-points 8 --max-candidates 24 --compute-fraction 0.75 --router-mode local --profile-cache runs/reproduce_fisher_pilot/fisher_profile.json --output-dir runs/reproduce_fisher_pilot/local
.venv/bin/python resnet_benchmark.py fisher-frontier --device cpu --batch-size 8 --budget-points 8 --max-candidates 24 --compute-fraction 0.75 --router-mode transfer --profile-cache runs/reproduce_fisher_pilot/fisher_profile.json --output-dir runs/reproduce_fisher_pilot/transfer
```

All eight ILP solves reported optimal and passed independent resource checks; they produced five unique seeds. Both modes share result caching, incumbent updates and early stopping. Only the transfer mode can propose edits borrowed from other budget contexts. It exercised that path twice in this run.

## Measured result

| Quantity | Local refinement | Feedback router |
|---|---:|---:|
| Unique candidate evaluations | 24 | 24 |
| Policy-image evaluations | 2392 | 2392 |
| Cross-budget transfer evaluations | 0 | 2 |
| Calibration + Fisher profiling (shared) (s) | 16.655 | 16.655 |
| Routing CPU time (s) | 0.033 | 0.026 |
| Search evaluation (s) | 100.813 | 105.886 |
| Held-out evaluation (s) | 186.704 | 195.926 |
| Actual pipeline wall time (s) | 308.345 | 306.138 |
| Wall time charging the shared profile to each run (s) | 308.345 | 322.793 |

“Shared profile” includes observer calibration plus Fisher profiling, charged once to each mode for comparison. Actual cache-hit wall time omits that work. These are single-run pipeline timings, excluding Python startup/imports; they do not establish a stable performance difference. The equal policy-image counts show no evaluation-work reduction in this experiment. Routing itself took less than 0.04 seconds in both modes.

| Estimated compression vs FP32 | Held-out top-1 | Origin |
|---:|---:|---|
| 6.9137× | 75.0% | Common ILP seed |
| 7.8724× | 57.8% | Common minimum-storage ILP seed |

FP32 accuracy was 73.0% on these 500 images. The observed 75.0% is an exploratory result on a small ten-class subset, not evidence of improved full-ImageNet accuracy. FP32 violates the experiment's BitOps cap and is shown as a separate reference. All sizes are estimates; execution uses fake quantization in FP32, not packed integer kernels.

[Router frontier](transfer/frontier.png) · [Local frontier](local/frontier.png) · [Machine-readable comparison](comparison.json)

## Correctness checks

- 22 unit checks passed across the existing runner and new implementation.
- Fisher scores agree with direct per-example autograd, including an in-place residual addition; unequal microbatches produce matching normalized scores and leave weights unchanged.
- Tiny exhaustive enumerations agree with MILP results under both resource constraints.
- Tests cover compatible transfers, failed-transfer context isolation, global deduplication, candidate limits, safe early rejection and KL tiebreaking.
- A real ResNet smoke run, the ILP-only command, and layer-policy export/reload completed. Reloaded policies reproduce every smoke-test prediction.
- Changing every smoke-test held-out label leaves selection unchanged. The selection artifact is written before held-out inference.
- The two pilot runs have identical profile fingerprints, ILP seeds, selected policies and held-out predictions.

The residual snapshot regression was found and fixed before these retained results were produced. Earlier development artifacts were moved outside the workspace. The profile fingerprint identifies the corrected scorer as `snapshot_v2`.

CUDA/MPS execution, full ImageNet, deployed integer latency and publishable novelty have not been established. The next research question is whether a stronger transfer proposal or more representative search data earns additional evaluations; this implementation does not assume that it does.
