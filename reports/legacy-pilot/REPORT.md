# ResNet-50 mixed-precision search: measured findings

**Status: completed pilot, 11 November 2026.** The initial signed-interaction surrogate did not establish an advantage over a simpler additive surrogate. The strongest verified result is exact search within a small, explicitly restricted stage-level policy space, accelerated by shared prefixes and safe early elimination. This is useful evidence, not proof of a new state-of-the-art method.

## Recommendation and actual policy

For this model and modest search space, start with **shared-prefix exact search**. Add safe elimination when seed profiling is already available or its cost is justified. The elimination version was about 2.36× faster than ordinary enumeration including estimated seed-selection cost, but only about 1.10× faster than shared-prefix enumeration. Three timing repetitions do not establish that the smaller advantage is robust.

At a budget of 85% of all-W8A8 model storage and 75% of its BitOps, the selected policy was:

| Component | Weights | Activations |
|---|---:|---:|
| Stem and classifier, fixed | 8 | 8 |
| ResNet layer1 / conv2_x | 8 | 8 |
| ResNet layer2 / conv3_x | 8 | 8 |
| ResNet layer3 / conv4_x | 4 | 8 |
| ResNet layer4 / conv5_x | 4 | 8 |

It uses an estimated **14,713,544 bytes** of packed weights, scales and biases, versus 25,723,592 for all-W8A8: **42.8% less storage**. Its estimated 188,953,395,200 BitOps are **27.8% lower** than all-W8A8. These are accounting estimates, not an exported integer model or measured integer inference speed. Activation working memory is not included in the storage objective.

| Policy | Correct / 500 | Test accuracy |
|---|---:|---:|
| Floating-point reference | 365 | 73.0% |
| All searchable stages W4A8 | 332 | 66.4% |
| All W8A8 | 365 | 73.0% |
| Selected mixed policy | 369 | 73.8% |

The selected-versus-float difference is +0.8 percentage points, with a paired, class-stratified bootstrap 95% interval of **−1.6 to +3.2 points**. There is no demonstrated improvement over floating point and no formal noninferiority test. Against all searchable stages W4A8, the difference is +7.4 points, interval **+4.2 to +10.6**. That comparison also gives the mixed policy more storage/compute; it demonstrates the benefit of selective upgrades, not dominance at equal resource use.

## How the method works

1. Fold batch normalization, calibrate activation ranges, and enumerate resource-feasible stage policies. Each of four stages has W4A8, W4A16, W8A8 or W8A16: 256 raw policies. Stem and classifier stay W8A8. The main budget leaves 20 policies.
2. Optionally score local joint quantization error on 50 profiling images. Fully evaluate the feasible policy with lowest summed stage MSE to obtain an incumbent with E mistakes on the 100 search images.
3. Order search images from largest to smallest floating-model cross entropy. This is a speed heuristic; correctness does not depend on the ordering.
4. Evaluate remaining policies in batches of eight. Represent policies as paths in a prefix tree. For the same image batch, policies sharing a stage prefix reuse exactly the same computed activations. Depth-first traversal avoids retaining every intermediate across the full dataset.
5. Stop evaluating a policy as soon as its observed mistake count exceeds E. Completely evaluate all surviving policies; select the fewest mistakes, breaking ties by mean teacher KL.

The implementation holds the initial E fixed. Updating it after a better completed policy would also be safe, but that improvement was not part of this measured implementation. This version needs **labeled search images**. An unlabeled variant could use nonnegative per-image teacher KL bounds, but its efficiency and accuracy have not been tested here.

## What is proved

Let P be the finite feasible policy set, and let m_p(t) be policy p's mistakes on the first t search images. Its full error count satisfies

`m_p(N) >= m_p(t)`.

A completely evaluated incumbent has E mistakes. If `m_p(t) > E`, then `m_p(N) > E`, so p cannot minimize full empirical error. Strict inequality preserves all possible error-count ties. All survivors complete evaluation and are compared using the same KL tie-breaker. Consequently, the returned policy is exactly optimal for **empirical search-set accuracy over P**, with the stated tie-breaker. Prefix reuse is exact because a deterministic stage's input depends only on its preceding policy prefix.

All nine saved elimination certificates were checked against independently completed exhaustive policy evaluations. Direct enumeration and prefix enumeration also produced identical classifications and numerically matching KL values in the timing experiment.

This does **not** prove optimal population accuracy, optimality over independently assigned per-layer precisions, or the fastest possible search algorithm. Without assumptions or valid lower bounds, an unevaluated policy can always be assigned a better arbitrary loss; a generic approximate surrogate cannot certify a global optimum. Restricting P makes an honest exact certificate practical.

## Evidence against the initial proposal

The original proposal profiled finite joint W/A errors and projected them through shared Fisher probes. The coupled score retains cross-stage inner products; the additive ablation drops those terms. Both were evaluated using the same quantizer, data and budget masks.

| Selector | Budget settings attaining best search accuracy / 9 |
|---|---:|
| Coupled signed-interaction score, one choice | 6 |
| Additive score, one choice | 8 |
| Local stage MSE, one choice | 3 |
| Coupled score, verify top 4 | 9 |
| Additive score, verify top 4 | 9 |
| Local MSE, verify top 4 | 5 |
| Shared-prefix exhaustive search | 9 |
| Shared-prefix search with safe elimination | 9 |

Budgets are the Cartesian product of storage fractions {0.70, 0.85, 0.95} and BitOps fractions {0.60, 0.75, 0.90}, relative to all-W8A8. Their union has 55 policies. These are correlated scenarios on one checkpoint and one split, **not nine independent replications**. The grid was fixed before measuring additional policies, after the first main-budget experiment. Top-four selectors succeeded here but have no general exactness guarantee.

The signed profiling pass itself took 10.33 seconds in the initial run, before policy verification and excluding its floating-teacher pass. That cold measurement is not a controlled timing comparison against the later warmed methods. It does establish that profiling overhead cannot be ignored. We did not implement or benchmark published HAWQ, BRECQ or other complete systems; no superiority claim over them is supported.

## Timing, including the missing cost

Same local Apple MPS device, 20 feasible policies, 100 search images, batches of eight, three repetitions:

| Operation | Median seconds | Min–max |
|---|---:|---:|
| Ordinary full-policy enumeration | 25.72 | 24.38–26.66 |
| Shared-prefix enumeration | 11.96 | 11.82–14.30 |
| Shared prefixes plus elimination | 8.27 | 7.88–9.32 |
| Standalone MSE seed profiling | 2.62 | 2.25–3.21 |
| Elimination plus seed profiling, estimated total | **10.88** | sum of separately measured medians |

The estimated total is not a separately timed end-to-end run. All methods exclude common loading, activation calibration, search-image preprocessing and floating-teacher inference. Calibration took 3.40 seconds in the initial run. No test-set evaluation cost is included in search timings. Timing order was fixed, not randomized; thermal/load effects can matter. The main racing run inspected 1,092 policy-image combinations instead of 2,000, with additional compute savings from shared prefixes.

## Data credibility and scope

We used the official [Torchvision ResNet-50 IMAGENET1K_V1 checkpoint and preprocessing](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet50.html), and the official [FastAI Imagenette 160-pixel archive](https://github.com/fastai/imagenette). Imagenette contains ten selected ImageNet classes. Checkpoint/archive SHA256 values and source URLs are in `sources.json`; each retained image's original archive path and SHA256 are in `manifest.json`.

Only 750 images are retained: 100 observer, 50 profile, 100 search, and 500 test. Splits are disjoint by file hash. The test set contains the 50 original `ILSVRC2012_val_*` images per class, recovered across both FastAI archive splits. Original ImageNet validation images never enter observer/profile/search. This matters because FastAI's train/validation repartition differs from ImageNet's original split. Predictions still compete across all 1,000 output classes.

The source images have a 160-pixel shorter side and are resized/cropped using the official 256/224 transform. This is **not full-resolution ImageNet-1K evaluation**. The ten classes, one split and small sample limit generalization. Bootstrap intervals condition on these ten classes and do not quantify uncertainty across other ImageNet classes. Held-out predictions were not used in policy scoring or selection, but the methodology was developed interactively after inspecting initial test results; a fresh external test is necessary for a confirmatory research claim.

Weights use per-output-channel symmetric max-absolute scales. Activations use calibrated per-tensor signed symmetric scales, including positive activations. There is no clipping optimization, reconstruction or fine-tuning. Biases, accumulators and residual addition remain floating point; fake quantization simulates only the specified tensor quantization. The folded float graph matched the original with maximum logit difference 1.15e-5 on the checked images. Actual integer export would require explicit treatment of addition, requantization and supported kernels.

Retained data/checkpoint occupy approximately 119 MiB on disk; the local environment adds about 24 MiB. The downloaded archive and unused partial/mirror files were removed. No additional download is needed to rerun the experiment.

## Reproduce and inspect

From `/Users/karankapur/amp`, using the existing environment:

For automatic policy selection alone, with no exhaustive-oracle or held-out-test evaluation:

```sh
.venv/bin/python resnet_benchmark.py search --memory-fraction 0.85 --compute-fraction 0.75
```

This writes `selected_policy.json` with stage precisions, resource use and an elimination certificate. A final standalone verification returned the same `[W8A8, W8A8, W4A8, W4A8]` policy: 9.20 seconds for profiling plus search, 12.07 seconds including loading, preprocessing, calibration and teacher inference inside the function. This is one additional run, not a replacement for the three-repetition timing comparison; Python import/startup time is excluded.

To reproduce the full benchmark and audit:

```sh
.venv/bin/python resnet_benchmark.py prepare
.venv/bin/python resnet_benchmark.py run
.venv/bin/python resnet_benchmark.py compare
.venv/bin/python resnet_benchmark.py seed-timing
.venv/bin/python analyze_resnet_results.py
```

`prepare` verifies the retained checkpoint and subset hashes without needing the removed archive. The script does not download missing assets automatically. GPU access may require execution outside the app sandbox; CPU fallback works but timings differ. Existing environment: Python 3.12, Torch 2.4.1, Torchvision 0.19.1, NumPy 2.0.2, SciPy 1.14.1. `compare` uses profiles from `run` and reuses saved test-policy measurements; for changed quantization/calibration settings use a fresh results directory to avoid stale caches.

`comparison.json` contains every budget's selections, timings and elimination certificate. `grid_oracle.npz` contains all 55 policies' per-image KL, cross entropy and mistakes. `audit.json` contains certificate checks, paired uncertainty and timing summaries. `analyze_resnet_results.py` reproduces those checks without GPU inference. The earlier `mixed_precision_strategy.md` records the research hypothesis; this report supersedes its untested recommendation.

## What remains unproven

The exact search machinery combines established ideas; **publication-level novelty has not been established**. The next research candidate is adaptive refinement from stages to residual blocks, using signed local scores only to propose refinements and exact empirical evaluation to accept them. This could keep search small while escaping coarse stage assignments, but its certificate would cover only the explored candidate set. It has not been implemented or tested and is not the recommendation supported by these measurements. Full layer-level, multi-checkpoint and full-distribution testing remains necessary before claiming a generally best mixed-precision policy search method.
