The 6.94× storage estimate is valid. The original 75.2% accuracy is reproducible on the original MPS arithmetic, but it is not a robust result, and there is no demonstrated improvement over FP32.

Audit completed against the retained pilot, with independent CPU inference, a real packed-weight round trip, fresh exhaustive MPS searches, and a rounding correction. The original files in `runs/pareto_pilot` remain historical evidence. This report supersedes interpreting that plot as a confirmed accuracy improvement or a deployment memory/speed result.

**The decisive finding: device-dependent quantization rounding**

On this machine's PyTorch 2.4.1, native `torch.round` uses different rules:

| Input | CPU | MPS |
|---|---:|---:|
| −2.5 | −2 | −3 |
| −0.5 | 0 | −1 |
| 0.5 | 0 | 1 |
| 2.5 | 2 | 3 |

PyTorch documents nearest rounding with ties to even. The original runner relied on native `.round()` for quantization. Its activation quantizers therefore implemented different tie rules on CPU and MPS. Weight quantization happened on CPU during construction, before moving the model to MPS. This was an activation arithmetic inconsistency, not secretly unquantized weights.

The correction adds explicit ties-to-even rounding on MPS and uses that helper for weight and activation quantization. CPU retains its native ties-to-even behavior. The GPU regression check covers **196,608 values**: all signed half-integers across the A16 range and the immediately adjacent representable values. All pass. Nine runner unit tests pass; the CPU-only unit run is supplemented by the actual MPS regression in `backend_fixed/backend_comparison.json`.

| Same 500 test images | Correct | Accuracy |
|---|---:|---:|
| Original torchvision FP32 checkpoint | 365 | 73.0% |
| Original policy `[2,3,0,1]`, original MPS rounding | 376 | 75.2% |
| Original policy, independent packed-weight CPU model | 371 | 74.2% |
| Original policy, corrected MPS rounding | 372 | 74.4% |
| Policy selected again with corrected rounding, `[3,3,0,0]`, MPS | 367 | 73.4% |
| Policy selected again with corrected rounding, `[3,3,0,0]`, CPU | 369 | 73.8% |

Both quantized policies in this table have the same 14,713,544-byte packed storage estimate and 6.940672× compression. **P04 is an ID local to a run:** the original and corrected frontiers assign that ID to different policies. Keeping the original policy fixed is distinct from repeating policy selection after correcting the arithmetic.

Corrected rounding does not guarantee identical CPU/GPU inference. The original policy still has five different top-1 predictions across CPU and corrected MPS. In a traced disagreement, all 50 quantizers give identical integer codes when supplied the same CPU input; upstream arithmetic differences can cross quantization boundaries and propagate. These discontinuities can make final logit changes appreciably larger than the initial floating-point differences. Native CPU/MPS round disagreement is a confirmed issue; it is not claimed to explain every remaining numerical discrepancy. Deployment kernels and backend must be validated explicitly.

Evidence: `backend_comparison.json` records the original native-rounding behavior; `backend_fixed/backend_comparison.json` records the correction and remaining differences. The original CPU/GPU comparison had nine different predictions. Both CPU implementations agree exactly on all 500 images, so the discrepancy is not an independent-implementation error.

**What was independently verified**

1. **Input integrity and split isolation.** All 750 retained image hashes match their manifest entries; all 750 encoded files and decoded RGB images are distinct. Class-folder WNIDs agree with the original 1,000-class label map. Every test filename is an original `ILSVRC2012_val_*` image; none appears in observer, profile, or search. Test rows and labels exactly match the predictions CSV. The checkpoint hash is `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.
2. **Preprocessing.** An independently written PIL resize, center crop, and normalization pipeline exactly matches the runner on all 500 test images. It uses the V1 checkpoint's 256 resize / 224 crop and RGB normalization. No class-restricted argmax is used: every prediction competes over all **1,000 logits**.
3. **FP32 reference.** The unmodified torchvision ResNet-50, loaded from the retained checkpoint, returns exactly the saved 365/500 predictions. The manually checked runner's folded FP32 graph also returns exactly those predictions. Its maximum logit difference from the original graph is **2.8610e−5** across the full test set. Evaluation mode is used; batch normalization is folded and there is no dropout, training, or label-dependent inference.
4. **Independent quantized model.** A second implementation keeps torchvision's graph, manually folds batch normalization, derives integer weights/scales with NumPy, packs and reloads the bytes, and inserts quantization through hooks. It does not use the runner's graph, shared-prefix traversal, loss function, or storage function for inference. All reconstructed quantized weights and all final CPU logits match the runner **exactly**. It gets 371/500 for the original policy.
5. **Actual packing.** A real raw payload was written and reloaded, including 4-bit nibbles, 8-bit weights, FP32 per-channel scales, FP32 biases, and 50 FP32 activation bounds. The counted size equals the original estimate exactly. This artifact is a storage-format verification, not an integer inference engine.
6. **Saved-score accounting.** Recomputing correctness from every stored prediction reproduces all seven plotted points. A fresh original-arithmetic MPS run at batch size 16 reproduces all **3,500** validation predictions and all **25,600** search error indicators from the original batch-size-8 run. Search KL/CE differ only numerically; candidate selection is identical.
7. **Search logic.** Independent reconstruction of the original frontier from the saved 256×100 measurements matches all four shortlisted policies. All 256 possible four-stage assignments are present. Candidates are selected using search errors, then teacher KL and resource tie-breaks; test labels are used only for evaluation and labeling the observed validation frontier. The corrected search was rerun from scratch, including calibration, enumeration and validation. Corrected batch sizes 8 and 16 give identical results for all 25,600 search error indicators and 3,500 validation predictions (`corrected_replay_comparison.json`).
8. **Calibration.** Observer flags are disabled for search/evaluation, and evaluation uses fixed saved bounds. Three diagnostic replacements of the 100 calibration images, drawn from the 150 observer/profile training images while excluding search/test, give 373, 375, and 372 correct under the original MPS arithmetic: **74.6%, 75.0%, 74.4%**. These are overlapping sensitivity trials, not independent replications, and were not used to choose a preferred result.

These checks found no evidence of a label-index bug, test-image duplication, test-driven candidate selection in the code, a broken FP32 baseline, stale frontier cache reuse, a class-restricted classifier, or a bypass around the weight quantizers. Exact encoded/decoded duplication checks do not rule out differently edited near-duplicates. The original full archive is no longer retained, so this audit verifies retained files against the recorded hashes rather than re-downloading the archive.

The earlier budget-grid/surrogate experiments under `resnet_results` also predate the rounding correction. This audit checks the Pareto pipeline and does not revalidate those separate method-ranking or timing claims.

**What “6.94× compression” actually means**

| Component | Bytes |
|---|---:|
| Packed mixed-precision weights | 14,492,864 |
| FP32 per-output-channel weight scales | 110,240 |
| FP32 biases | 110,240 |
| 50 FP32 activation bounds/scales | 200 |
| **Quantized payload total** | **14,713,544** |
| **Folded FP32 weights and biases** | **102,121,888** |

`102,121,888 / 14,713,544 = 6.940672349`. This is about **85.59% less parameter payload storage**. The numerator and denominator both account for the folded model; the numerator is not the checkpoint archive's disk size. Separate architecture/layout metadata and container overhead are not included. The raw payload's format is documented in `packing.json`.

There are 22,020,096 four-bit weights and 3,482,816 eight-bit weights: approximately 86.3% of weights are four-bit. Thus a result close to the ideal eightfold FP32-to-INT4 weight reduction is arithmetically unsurprising. Stem and classifier stay eight-bit. Four-bit weights use 15 symmetric levels, −7 through +7, with one FP32 scale per output channel; they are genuinely restricted to that grid.

However, the runner's resident model tensors occupy **306,145,384 bytes**, because it retains original FP32 weights and dequantized FP32 copies of both W4 and W8. This excludes activations, allocator/workspace memory and temporary tensors. The current simulator therefore does **not** deliver 6.94× lower resident RAM/VRAM. The independent packed payload can be stored compactly, but its CPU inference test expands weights to FP32 again.

Convolutions, linear operations, bias, accumulation, residual additions, ReLUs, pooling and logits execute in FP32. Activation codes use A8/A16, then are dequantized to FP32. A16 means signed 16-bit integer quantization, not FP16 arithmetic. Activation working memory is omitted from the compression axis. No latency, energy, integer overflow/requantization behavior, or deployable integer-kernel accuracy claim is supported.

The original policy's estimated BitOps are **1.07225× all-W8A8**, or 7.2% higher. The corrected search's same-size selected policy uses **1.13664× all-W8A8**, or 13.7% higher. The frontier optimizes parameter storage and empirical accuracy, not speed or activation memory. Cheap-looking points can spend more activation precision to preserve accuracy.

**What the accuracy numbers can support**

The original gain of +2.2 percentage points is 26 corrected FP32 mistakes versus 15 new mistakes. Its paired exact McNemar p-value is **0.1173**. A 20,000-draw class-stratified paired bootstrap gives **−0.2 to +4.6 percentage points**. Even before the rounding correction, that does not establish superiority. The nominal Wilson interval for the original 75.2% is **71.2–78.8%**, conditional on this restricted evaluation population; it does not establish that true accuracy exceeds 74% or 75%.

| Comparison against FP32 | Observed difference | Paired 95% bootstrap interval | Exact McNemar p |
|---|---:|---:|---:|
| Original policy, original MPS | +2.2 pp | −0.2 to +4.6 pp | 0.1173 |
| Original policy, CPU | +1.2 pp | −1.4 to +3.6 pp | 0.4408 |
| Original policy, corrected MPS | +1.4 pp | −1.0 to +3.8 pp | 0.3489 |
| Reselected policy, corrected MPS | +0.4 pp | −2.2 to +3.0 pp | 0.8804 |
| Reselected policy, CPU | +0.8 pp | −1.8 to +3.4 pp | 0.6587 |

These are **exploratory, nominal intervals**, not multiplicity-adjusted confirmatory claims. The test set has been inspected across several methods and policies. Failure to show a difference is not proof of equivalence or noninferiority. For example, the corrected selected policy's interval still permits a loss larger than two percentage points on this restricted distribution.

The original policy is also not a stable unique winner of its 100-image search sample: in 2,000 class-stratified bootstrap resamples, it wins among the 16 same-storage policies 1,085 times (54.25%). This is a search-stability diagnostic, not additional test evidence. Three of the four search-frontier policy assignments change after correcting the rounding rule.

Class behavior is uneven. The original P04 improves English springer from 37 to 42 correct, but reduces church from 18 to 14 and golf ball from 41 to 38. All classes contain 50 test images. A class-resampling sensitivity calculation for the original improvement spans −1.6 to +5.6 pp, but these ten chosen classes are not a random sample of ImageNet, so that is not an ImageNet-wide confidence interval.

**The largest scope limitations beyond “only 500 images”**

- **Ten deliberately selected classes.** These are Imagenette classes, described by its authors as easily classified. The classifier still has 1,000 outputs, but evaluation covers only 1% of classes. Calibration and policy search use those same ten classes. No conclusion about the other 990 classes follows from this experiment.
- **Reduced source resolution.** All retained files have 160-pixel short sides before upsampling to the standard 256/224 preprocessing. This is not the original-resolution ImageNet benchmark. The official V1 FP32 ImageNet-1K accuracy, 76.13%, is not a directly comparable baseline for these files.
- **Repeatedly inspected validation data.** The software separates splits correctly, but the research process has inspected these results repeatedly. The plotted “validation frontier” is selected using these validation outcomes and is exploratory. Fixing a bug and examining the same sample again does not create a fresh confirmatory experiment.
- **Small, coarse search space.** Enumeration is exact for 256 stage assignments on 100 search images. It is not globally optimal per-layer quantization or a population-accuracy guarantee. Only shortlisted policies and two anchors are evaluated on validation.
- **No deployment evidence and no novelty proof.** Low-bit ResNet post-training quantization is an established field. A plausible four-bit storage/accuracy trade-off does not alone establish a new method, especially without matched full-distribution baselines and actual integer execution.

**Research decision**

This is worth a controlled follow-up as a promising storage/accuracy pilot. It is not evidence of a breakthrough or of beating FP32. The strongest defensible statement is:

> On 500 downsampled images from ten ImageNet classes, a mixed-precision ResNet-50 policy has a verified 14.7 MB packed parameter payload versus 102.1 MB folded FP32. With explicit ties-to-even quantization and a repeated search, it scores 73.4% on MPS versus 73.0% FP32. Accuracy improvement, noninferiority, full-ImageNet generalization, and deployment performance remain unestablished.

For a confirmatory follow-up, freeze the rounding rule, checkpoint, calibration procedure, policy and comparison before evaluation. Evaluate full-resolution images across all 1,000 classes, with paired predictions, a declared acceptable accuracy-loss margin, and the same backend for both models. The standard ImageNet-1K validation set has **50,000 images, 50 per class**. Its original 500 pilot identities have already been inspected, so report that overlap and separately evaluate the untouched 49,500 if using that set. No full ImageNet dataset is available here; none was downloaded or silently substituted. Before claiming runtime savings, implement and test the intended integer backend, including residual addition and requantization.

**Artifacts and reproduction**

- `data_integrity.json`, `statistics.json`, `search_audit.json`: retained data, original statistics and independent search checks.
- `independent_inference.json`, `independent_predictions.csv`, `cpu_P04_logits.npz`: CPU graph and numerical checks.
- `P04_packed_payload.bin`, `packing.json`: actual packed original-policy weights and layout.
- `replay_mps/`, `replay_comparison.json`: original-arithmetic MPS replay at batch size 16.
- `backend_comparison.json`: original CPU/MPS rounding trace, including the eight exact half-integer examples.
- `backend_fixed/`: corrected fixed-policy MPS logits and arithmetic trace; 196,608-case MPS regression.
- `corrected_mps/`, `corrected_mps_batch8/`: full corrected searches and validation at two batch sizes. The second run records explicit rounding metadata; the first began before that metadata field was added.
- `corrected_selected_cpu/`: CPU evaluation of the policy selected by the corrected MPS search.
- `accuracy_comparisons.json`, `audit_summary.png`, `audit_summary.svg`: comparisons and uncertainty plot.
- `calibration_sensitivity/`: original-rounding calibration diagnostics; not reselected or used as the final result.
- `unit_tests.log`: nine runner unit tests.

```sh
# Independent retained-pilot checks, CPU inference, actual packing and summary:
.venv/bin/python audit_pareto.py

# Corrected GPU trace and regression (requires MPS access):
.venv/bin/python audit_backend.py

# Optional reproduction of the historical rounding discrepancy:
.venv/bin/python audit_backend.py --legacy-rounding --output-dir runs/pareto_audit/legacy_reproduced

# Corrected full search in a fresh directory:
.venv/bin/python resnet_benchmark.py frontier --device mps --batch-size 8 --output-dir runs/pareto_confirm

.venv/bin/python -m unittest test_resnet_runner -v
```

`audit_pareto.py` audits the original stored search data while independently rerunning its fixed policy on CPU. `resnet_benchmark.py frontier` performs a fresh search. `audit_backend.py --legacy-rounding` is explicitly historical; it is not the corrected default. Original saved policies without a rounding field now emit a warning when evaluated under corrected semantics.

Primary references: [Torchvision V1 checkpoint and preprocessing](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet50.html), [Imagenette source and resolution](https://github.com/fastai/imagenette), [ImageNet-1K dataset splits](https://huggingface.co/datasets/ILSVRC/imagenet-1k), [PyTorch rounding specification](https://docs.pytorch.org/docs/2.14/generated/torch.round.html), [PyTorch numerical accuracy limitations](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html). Prior low-bit PTQ work includes [AdaRound](https://arxiv.org/abs/2004.10568) and [BRECQ](https://arxiv.org/abs/2102.05426); no matched reproduction or superiority comparison with those systems was performed here.
