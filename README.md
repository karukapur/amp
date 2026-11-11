# ResNet-50 mixed-precision policy runner

**Audit correction:** the historical `runs/pareto_pilot` result of 75.2% at 6.94× storage compression depends on PyTorch 2.4.1 MPS using a different half-tie rounding rule from CPU. The runner now explicitly uses ties-to-even. The original policy gives 74.4% on corrected MPS and 74.2% on CPU; repeating the search selects a different same-size policy with 73.4% versus FP32's 73.0%. None demonstrates an accuracy improvement. The 6.94× packed parameter size was verified by writing and reloading real packed bytes, but runtime tensors and computation remain FP32. See the [full independent audit](runs/pareto_audit/REPORT.md) and [corrected results](runs/pareto_audit/corrected_mps_batch8/frontier.csv). Original pilot artifacts are preserved.

Automatically select stage precisions from W4A8 / W4A16 / W8A8 / W8A16, export the assignment for **every weight layer and activation quantizer**, and evaluate it against the floating-point model. Stem and classifier stay W8A8. Precision choices remain stage-wide: the detailed CSV expands those choices to each operation; it does not imply independent per-layer optimization.

## Recommended: choose from a Pareto frontier

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

`--device auto` prefers CUDA, then Apple MPS, then CPU. `--device cuda:1` selects a specific GPU. Explicit CUDA requests fail if CUDA is unavailable, instead of silently running on CPU. Reduce `--batch-size` if GPU memory is tight. Local/Fisher profiling retains its smaller batch size of four. No distributed/multi-GPU execution is implemented.

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
