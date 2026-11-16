"""Isolated v2 runner with group repair and editable layer precision formats.

The Fisher command chooses precision independently for all 54 operators. Legacy
commands retain four stage groups and fixed W8A8 stem/classifier behavior.
Execution is FP32 fake quantization; see docs/group-repair-v2.md for v2 commands.
"""
import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import shlex
import sys
import tarfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from torchvision.models import resnet50, ResNet50_Weights

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark_data"
OUT = ROOT / "resnet_results"
MODES = ((4, 8), (4, 16), (8, 8), (8, 16))
WNIDS = ("n01440764", "n02102040", "n02979186", "n03000684", "n03028079", "n03394916", "n03417042", "n03425413", "n03445777", "n03888257")
TRANSFORM = ResNet50_Weights.IMAGENET1K_V1.transforms()
CHECKPOINT = DATA / "resnet50-0676ba61.pth"
MANIFEST = OUT / "manifest.json"
IMAGE_ROOT = ROOT
DEVICE = "auto"
BATCH_SIZE = 8
PROFILE_BATCH_SIZE = 4
LOGGER = logging.getLogger("amp.runner_v2")


def log(message):
    LOGGER.info(message)


@contextmanager
def run_logging(output_dir, phase):
    """Flush application progress, Python warnings and failures to console and disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "main.log"
    formatter = logging.Formatter("%(asctime)s.%(msecs)03dZ | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    handlers = [logging.FileHandler(path, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    previous_level, previous_propagate = LOGGER.level, LOGGER.propagate
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in handlers:
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    start = perf_counter()
    try:
        with warnings.catch_warnings():
            def show_warning(message, category, filename, lineno, file=None, line=None):
                LOGGER.warning(warnings.formatwarning(message, category, filename, lineno, line).rstrip())
            warnings.showwarning = show_warning
            log(f"[run] Started phase={phase}, pid={os.getpid()}; timestamps are UTC.")
            log(f"[run] Live log: {path.resolve()}")
            try:
                yield
            except KeyboardInterrupt:
                LOGGER.warning("[run] Interrupted after %.2fs; partial outputs may remain.", perf_counter() - start)
                raise SystemExit(130) from None
            except SystemExit as exc:
                LOGGER.log(logging.ERROR if exc.code else logging.INFO,
                           "[run] Exited status=%s after %.2fs.", exc.code, perf_counter() - start)
                raise
            except Exception:
                LOGGER.exception("[run] Failed after %.2fs.", perf_counter() - start)
                raise SystemExit(1) from None
            else:
                log(f"[run] Completed phase={phase} in {perf_counter() - start:.2f}s. Outputs: {output_dir.resolve()}")
    finally:
        for handler in handlers:
            LOGGER.removeHandler(handler)
            handler.close()
        LOGGER.setLevel(previous_level)
        LOGGER.propagate = previous_propagate


class RunArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        LOGGER.error("[arguments] %s", message)
        super().error(message)


def select_device(requested=None):
    requested = requested or DEVICE
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
            raise ValueError("Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1: this run requires an FP32 reference without forced TF32.")
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable. Install CUDA-enabled PyTorch and check the NVIDIA driver.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist.")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        # Preserve FP32 reference arithmetic rather than silently enabling TF32.
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable.")
    elif device.type not in ("cpu", "mps"):
        raise ValueError("Supported devices: auto, cpu, mps, cuda, cuda:N.")
    return device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_runtime(device, check=False):
    """Report the selected logical GPU and optionally exercise forward/backward kernels."""
    device = torch.device(device)
    result = {"device": str(device), "torch_version": str(torch.__version__)}
    if device.type != "cuda":
        return result
    properties = torch.cuda.get_device_properties(device)
    result.update(gpu_name=properties.name, compute_capability=[properties.major, properties.minor],
                  total_memory_bytes=properties.total_memory, torch_cuda_version=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(), fp32_precision="ieee")
    if check:
        try:
            with torch.enable_grad():
                x = torch.ones(2, 3, 8, 8, device=device, requires_grad=True)
                output = F.conv2d(x, torch.ones(4, 3, 3, 3, device=device) / 27)
                # Exercise quantization, cuDNN, cuBLAS and input-gradient propagation.
                quantized = output.detach().clamp(-2, 2).mul(127).round().div(127)
                logits = F.linear(output.flatten(1), torch.ones(5, 144, device=device) / 144)
                logits.square().mean().backward()
                if not torch.isfinite(x.grad).all() or not torch.isfinite(quantized).all():
                    raise RuntimeError("Non-finite CUDA preflight result.")
                sync(device)
        except torch.cuda.OutOfMemoryError as exc:
            raise torch.cuda.OutOfMemoryError(f"CUDA preflight ran out of memory on {device}; free GPU memory before starting.") from exc
        except RuntimeError as exc:
            raise RuntimeError(f"CUDA forward/backward preflight failed on {device} ({properties.name}). Check the driver and install a matched CUDA-enabled Torch/Torchvision build supporting this GPU. Original error: {exc}") from exc
        result["forward_backward_check"] = "passed"
    return result


def prepare():
    OUT.mkdir(exist_ok=True)
    digest = hashlib.sha256((DATA / "resnet50-0676ba61.pth").read_bytes()).hexdigest()
    assert digest == "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
    if (OUT / "manifest.json").exists():
        manifest = read_manifest()
        for rows in manifest.values():
            for row in rows:
                assert hashlib.sha256((ROOT / row["file"]).read_bytes()).hexdigest() == row["sha256"]
        log("Verified retained subset and checkpoint; no archive needed.")
        return
    assert hashlib.sha256((DATA / "imagenette2-160.tgz").read_bytes()).hexdigest() == "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
    labels = {v[0]: int(k) for k, v in json.loads((DATA / "imagenet_class_index.json").read_text()).items()}
    rng = np.random.default_rng(20261111)
    manifest = {name: [] for name in ("observer", "profile", "search", "test")}
    with tarfile.open(DATA / "imagenette2-160.tgz") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith((".jpeg", ".jpg")) and "/._" not in m.name]
        chosen_members = []
        for wnid in WNIDS:
            group = sorted((m for m in members if Path(m.name).parent.name == wnid), key=lambda m: m.name)
            original_val = [m for m in group if Path(m.name).name.startswith("ILSVRC2012_val_")]
            original_train = [m for m in group if not Path(m.name).name.startswith("ILSVRC2012_val_")]
            rng.shuffle(original_train)
            assert len(original_val) == 50, (wnid, len(original_val))
            assignments = {"observer": original_train[:10], "profile": original_train[10:15], "search": original_train[15:25], "test": original_val}
            for split, selected in assignments.items():
                directory = DATA / "subset" / split
                directory.mkdir(parents=True, exist_ok=True)
                for member in selected:
                    target = directory / Path(member.name).name
                    chosen_members.append((member, target, split, wnid))
        # Read selected members in archive order to avoid repeated gzip seeks.
        for member, target, split, wnid in sorted(chosen_members, key=lambda row: row[0].offset):
            payload = archive.extractfile(member).read()
            target.write_bytes(payload)
            manifest[split].append({"file": str(target.relative_to(ROOT)), "label": labels[wnid], "wnid": wnid, "sha256": hashlib.sha256(payload).hexdigest(), "archive_member": member.name})
        for rows in manifest.values():
            rng.shuffle(rows)
    names = [[r["file"].split("/")[-1] for r in rows] for rows in manifest.values()]
    assert len(set(itertools.chain.from_iterable(names))) == sum(map(len, names))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sources = {
        "checkpoint": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "checkpoint_sha256": digest,
        "dataset": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
        "dataset_sha256": hashlib.sha256((DATA / "imagenette2-160.tgz").read_bytes()).hexdigest(),
        "dataset_author": "https://github.com/fastai/imagenette",
        "test_members": "Only original ILSVRC2012 validation filenames; no original ImageNet validation images enter calibration or search.",
        "resolution": "FastAI 160px source images; official torchvision resize-to-256/center-crop-224 preprocessing. This is not the full-resolution ImageNet benchmark.",
        "split_sizes": {k: len(v) for k, v in manifest.items()},
    }
    (OUT / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    log(sources)


def load_images(rows):
    images = []
    for row in rows:
        path = IMAGE_ROOT / row["file"]
        try:
            with Image.open(path) as im:
                images.append(TRANSFORM(im.convert("RGB")))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot decode image {path}: {exc}") from exc
    return torch.stack(images), torch.tensor([r["label"] for r in rows])


def read_manifest():
    manifest = json.loads(MANIFEST.read_text())
    seen = set()
    for split, rows in manifest.items():
        if split not in ("observer", "profile", "search", "test") or not isinstance(rows, list):
            raise ValueError(f"Invalid manifest split: {split}")
        for row in rows:
            if type(row.get("label")) is not int or not 0 <= row["label"] < 1000:
                raise ValueError("Labels must be original ImageNet indices from 0 to 999.")
            path = (IMAGE_ROOT / row["file"]).resolve()
            if path in seen:
                raise ValueError(f"Image appears more than once across splits: {path}")
            if not path.is_file():
                raise ValueError(f"Missing image: {path}")
            seen.add(path)
    for split in ("observer", "profile", "search"):
        if not manifest.get(split):
            raise ValueError(f"Manifest requires a nonempty {split} split.")
    return manifest


def imagenet_manifest(root, counts, seed):
    """Sample disjoint training images; retain all class-organized validation paths."""
    root = root.expanduser().resolve()
    log(f"[dataset] Scanning ImageNet at {root}; requested observer/profile/search={counts}, seed={seed}.")
    index_path = DATA / "imagenet_class_index.json"
    if not index_path.is_file():
        raise ValueError(f"Missing ImageNet label mapping: {index_path}. Copy benchmark_data/imagenet_class_index.json to the remote server alongside the checkpoint.")
    index = json.loads(index_path.read_text())
    if not isinstance(index, dict) or set(index) != {str(i) for i in range(1000)} or any(not isinstance(v, list) or len(v) != 2 or not isinstance(v[0], str) for v in index.values()):
        raise ValueError("ImageNet label mapping must contain the original 1,000 class indices.")
    labels = {v[0]: int(k) for k, v in index.items()}
    if len(labels) != 1000:
        raise ValueError("ImageNet label mapping contains duplicate class folders.")
    if (root / "train").resolve() == (root / "val").resolve():
        raise ValueError("ImageNet train and val must be separate directories.")
    directories = {}
    for split in ("train", "val"):
        if not (root / split).is_dir():
            raise ValueError(f"Expected {root / split}/<ImageNet WNID>/*.JPEG")
        directories[split] = sorted(p for p in (root / split).iterdir() if p.is_dir())
        if not directories[split] or any(p.name not in labels for p in directories[split]):
            raise ValueError(f"{split} must contain class folders named by ImageNet WNIDs (e.g. n01440764).")
    rng = np.random.default_rng(seed)
    pools = {}
    manifest = {name: [] for name in ("observer", "profile", "search", "test")}
    def files(directory):
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg", ".png") and not p.name.startswith("._"))
    for split, count in zip(("observer", "profile", "search"), counts):
        if count <= 0:
            raise ValueError("Observer, profile and search image counts must be positive.")
        while len(manifest[split]) < count:
            progress = False
            for i in rng.permutation(len(directories["train"])):
                directory = directories["train"][i]
                if directory not in pools:
                    pools[directory] = files(directory)
                    rng.shuffle(pools[directory])
                    if len(pools) == 1 or len(pools) % 100 == 0 or len(pools) == len(directories["train"]):
                        log(f"[dataset] Indexed training folders {len(pools)}/{len(directories['train'])}.")
                if pools[directory]:
                    path = pools[directory].pop()
                    if path.name.startswith("ILSVRC2012_val_"):
                        raise ValueError("Original validation image found in train; fix the ImageNet split.")
                    manifest[split].append({"file": str(path), "label": labels[directory.name]})
                    progress = True
                if len(manifest[split]) == count:
                    break
            if not progress:
                raise ValueError("Not enough distinct training images for the requested splits.")
        log(f"[dataset] Selected {len(manifest[split])} disjoint {split} images.")
    for index, directory in enumerate(directories["val"], 1):
        manifest["test"].extend({"file": str(p), "label": labels[directory.name]} for p in files(directory))
        if index == 1 or index % 100 == 0 or index == len(directories["val"]):
            log(f"[dataset] Indexed validation folders {index}/{len(directories['val'])}; {len(manifest['test'])} images so far.")
    if not manifest["test"]:
        raise ValueError("No validation images found.")
    return manifest


def round_to_even(x):
    """Use one quantizer tie rule even on MPS versions with away-from-zero round."""
    if x.device.type != "mps":
        return x.round()
    # PyTorch 2.4.1 MPS round(0.5)==1, unlike CPU/CUDA's documented ties-to-even.
    lower = x.floor()
    fraction = x - lower
    up = (fraction > 0.5) | ((fraction == 0.5) & (lower.remainder(2) != 0))
    return lower + up.to(x.dtype)


class Activation(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bound", torch.tensor(1e-5))
        self.observe = False

    def forward(self, x, bits):
        if self.observe:
            self.bound.copy_(torch.maximum(self.bound, x.detach().abs().amax()))
        if bits is None:
            return x
        scale = self.bound / (2 ** (bits - 1) - 1)
        clipped = x.clamp(-self.bound, self.bound)
        rounded = round_to_even(clipped / scale) * scale
        if x.requires_grad:
            return clipped + (rounded - clipped).detach()
        return rounded


class Affine(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
        self.op.requires_grad_(False)
        self.count = op.weight.numel()
        self.macs = 0
        weight = op.weight.detach()
        bound = weight.flatten(1).abs().amax(dim=1).reshape((-1,) + (1,) * (weight.ndim - 1))
        for bits in (4, 8):
            scale = bound.clamp_min(1e-10) / (2 ** (bits - 1) - 1)
            self.register_buffer(f"w{bits}", round_to_even(weight / scale).clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1) * scale)

    def forward(self, x, bits):
        weight = self.op.weight if bits is None else getattr(self, f"w{bits}")
        if isinstance(self.op, nn.Conv2d):
            y = F.conv2d(x, weight, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
            self.macs = self.count * y.shape[-1] * y.shape[-2]
        else:
            y = F.linear(x, weight, self.op.bias)
            self.macs = self.count
        return y


def fused(conv, bn):
    return Affine(torch.nn.utils.fuse_conv_bn_eval(conv, bn))


class Block(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.input = Activation()
        self.a2 = Activation()
        self.a3 = Activation()
        self.c1 = fused(block.conv1, block.bn1)
        self.c2 = fused(block.conv2, block.bn2)
        self.c3 = fused(block.conv3, block.bn3)
        self.projection = fused(block.downsample[0], block.downsample[1]) if block.downsample is not None else None

    def forward(self, x, mode):
        w, a = (None, None) if mode is None else MODES[mode]
        x = self.input(x, a)
        y = F.relu(self.c1(x, w))
        y = F.relu(self.c2(self.a2(y, a), w))
        y = self.c3(self.a3(y, a), w)
        shortcut = x if self.projection is None else self.projection(x, w)
        return F.relu(y + shortcut)


class Stage(nn.Module):
    def __init__(self, stage):
        super().__init__()
        self.blocks = nn.ModuleList([Block(b) for b in stage])

    def forward(self, x, mode):
        for block in self.blocks:
            x = block(x, mode)
        return x


class QuantResNet(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.input = Activation()
        self.stem_op = fused(original.conv1, original.bn1)
        self.stages = nn.ModuleList([Stage(getattr(original, f"layer{i}")) for i in range(1, 5)])
        self.head_input = Activation()
        self.head_op = Affine(original.fc)

    def stem(self, x, floating=False):
        return F.max_pool2d(F.relu(self.stem_op(self.input(x, None if floating else 8), None if floating else 8)), 3, 2, 1)

    def head(self, x, floating=False):
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.head_op(self.head_input(x, None if floating else 8), None if floating else 8)

    def forward(self, x, policy=None, retain=False):
        x = self.stem(x, policy is None)
        inputs, outputs = [], []
        for i, stage in enumerate(self.stages):
            inputs.append(x)
            x = stage(x, None if policy is None else int(policy[i]))
            outputs.append(x)
        logits = self.head(x, policy is None)
        return (logits, inputs, outputs) if retain else logits


def validate_policy(policy):
    if not isinstance(policy, (list, tuple, np.ndarray)) or len(policy) != 4 or any(type(m) not in (int, np.int64, np.int32) or not 0 <= m < 4 for m in policy):
        raise ValueError("Policy must contain four mode IDs: 0=W4A8, 1=W4A16, 2=W8A8, 3=W8A16.")
    return tuple(int(m) for m in policy)


def export_policy_csv(model, policy, path):
    """Describe actual quantization boundaries, including floating graph operations."""
    policy = validate_policy(policy)
    rows = []
    def row(name, kind, stage, bits="", **values):
        rows.append(dict(name=name, kind=kind, stage=stage, quant_bits=bits,
                         compute_dtype="float32", **values))
    def activation(name, module, stage, bits):
        qmax = 2 ** (bits - 1) - 1
        row(name, "activation_quantizer", stage, bits, activation_bits=bits,
            scheme="symmetric_signed_maxabs", granularity="per_tensor",
            quant_min=-qmax, quant_max=qmax, scale=float(module.bound.cpu()) / qmax,
            notes="Round nearest, ties to even; fake quantize then dequantize to FP32; calibrated scale.")
    def affine(name, module, stage, w, a, input_name, bn_name=None):
        qmax = 2 ** (w - 1) - 1
        row(name, "weight", stage, w, weight_bits=w, activation_bits=a,
            bias_bits=32, accumulator_bits=32, scheme="symmetric_signed_maxabs",
            granularity="per_output_channel", quant_min=-qmax, quant_max=qmax,
            input_quantizer=input_name, parameter_count=module.count,
            notes="Quantized folded weights; bias and accumulation are FP32.")
        if bn_name:
            row(bn_name, "batchnorm_folded", stage, notes=f"Folded into {name}; no standalone inference operation.")
    activation("input_quantizer", model.input, "fixed", 8)
    affine("conv1", model.stem_op, "fixed", 8, 8, "input_quantizer", "bn1")
    row("relu", "activation_float", "fixed", notes="FP32 ReLU output; quantized only at the next explicit quantizer.")
    row("maxpool", "pool_float", "fixed")
    for i, stage in enumerate(model.stages):
        w, a = MODES[policy[i]]
        group = f"layer{i + 1}"
        for j, block in enumerate(stage.blocks):
            prefix = f"{group}.{j}"
            shared = f"{prefix}.input_quantizer"
            activation(shared, block.input, group, a)
            for k, (op, act) in enumerate(((block.c1, None), (block.c2, block.a2), (block.c3, block.a3)), 1):
                input_name = shared if k == 1 else f"{prefix}.conv{k}.input_quantizer"
                if act is not None:
                    activation(input_name, act, group, a)
                affine(f"{prefix}.conv{k}", op, group, w, a, input_name, f"{prefix}.bn{k}")
                if k < 3:
                    row(f"{prefix}.relu_after_conv{k}", "activation_float", group,
                        notes="FP32 ReLU output; next convolution input has an explicit quantizer.")
            if block.projection is not None:
                affine(f"{prefix}.downsample.0", block.projection, group, w, a, shared, f"{prefix}.downsample.1")
            row(f"{prefix}.residual_add", "addition_float", group, accumulator_bits=32,
                notes="Shortcut shares the block input quantizer; residual addition itself is FP32.")
            row(f"{prefix}.relu_out", "activation_float", group,
                notes="FP32 output; quantized at the next block input or classifier input after pooling.")
    row("avgpool", "pool_float", "fixed")
    activation("fc.input_quantizer", model.head_input, "fixed", 8)
    affine("fc", model.head_op, "fixed", 8, 8, "fc.input_quantizer")
    row("fc.output", "activation_float", "fixed", notes="FP32 classification logits; no output quantizer.")
    fields = ["name", "kind", "stage", "quant_bits", "weight_bits", "activation_bits", "bias_bits", "accumulator_bits", "compute_dtype", "scheme", "granularity", "quant_min", "quant_max", "scale", "input_quantizer", "parameter_count", "notes"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def losses(logits, teacher, labels):
    # Nonnegative finite per-image KL; rounding-sized negatives are set to zero.
    kl = (teacher * (teacher.clamp_min(1e-30).log() - logits.log_softmax(-1))).sum(-1).clamp_min(0)
    ce = F.cross_entropy(logits, labels, reduction="none")
    errors = logits.argmax(-1).ne(labels)
    return torch.stack((kl, ce, errors.float()), dim=1)


def batch_forward(model, x, policy, device, batch=None):
    batch = batch or BATCH_SIZE
    values = []
    with torch.no_grad():
        for part in x.split(batch):
            values.append(model(part.to(device), policy).cpu())
    sync(device)
    return torch.cat(values)


def costs(model, policies):
    weight_count = np.array([sum(m.count for m in stage.modules() if isinstance(m, Affine)) for stage in model.stages])
    mac_count = np.array([sum(m.macs for m in stage.modules() if isinstance(m, Affine)) for stage in model.stages])
    fixed_weights = model.stem_op.count + model.head_op.count
    fixed_macs = model.stem_op.macs + model.head_op.macs
    weight_bits = np.array([MODES[s][0] for s in range(4)])[policies]
    activation_bits = np.array([MODES[s][1] for s in range(4)])[policies]
    overhead = sum(4 * (m.op.weight.shape[0] + (m.op.bias.numel() if m.op.bias is not None else 0)) for m in model.modules() if isinstance(m, Affine))
    overhead += 4 * sum(isinstance(m, Activation) for m in model.modules())
    size = (weight_bits * weight_count).sum(-1) / 8 + fixed_weights + overhead
    bitops = (weight_bits * activation_bits * mac_count).sum(-1) + 64 * fixed_macs
    ref_size = weight_count.sum() + fixed_weights + overhead
    ref_bitops = 64 * (mac_count.sum() + fixed_macs)
    # Fixed before observing any quantized accuracy. Overhead is held constant.
    feasible = (size <= 0.85 * ref_size) & (bitops <= 0.75 * ref_bitops)
    return size, bitops, feasible, {"weight_budget_bytes": 0.85 * ref_size, "bitops_budget": float(0.75 * ref_bitops), "all_w8a8_bytes": float(ref_size), "all_w8a8_bitops": float(ref_bitops), "stage_weight_counts": weight_count.tolist(), "stage_macs": mac_count.tolist(), "constant_bias_scale_bytes": overhead}


def profile(model, x, teacher, device, probes=4):
    n = len(x)
    linear = torch.zeros(4, 4)
    signatures = torch.zeros(4, 4, n, probes)
    local_mse = torch.zeros(4, 4)
    start = perf_counter()
    generator = torch.Generator().manual_seed(2718)
    for offset in range(0, n, 4):
        part = x[offset:offset + 4].to(device).requires_grad_(True)
        target = teacher[offset:offset + 4].to(device)
        logits, inputs, outputs = model(part, (3, 3, 3, 3), retain=True)
        p = logits.detach().softmax(-1)
        errors = []
        with torch.no_grad():
            for i in range(4):
                e = torch.stack([model.stages[i](inputs[i].detach(), mode) - outputs[i].detach() for mode in range(4)])
                errors.append(e)
                local_mse[i] += e.square().flatten(2).mean(-1).sum(-1).cpu() / n
        loss = (target * (target.clamp_min(1e-30).log() - logits.log_softmax(-1))).sum() / n
        gradients = torch.autograd.grad(loss, outputs, retain_graph=True)
        for i in range(4):
            linear[i] += (errors[i] * gradients[i].unsqueeze(0)).flatten(1).sum(-1).cpu()
        for r in range(probes):
            noise = torch.randn(p.shape, generator=generator).to(device) * p.sqrt()
            q = (noise - p * noise.sum(-1, keepdim=True)).detach()
            gradients = torch.autograd.grad((q * logits).sum(), outputs, retain_graph=r < probes - 1)
            for i in range(4):
                signatures[i, :, offset:offset + len(part), r] = (errors[i] * gradients[i].unsqueeze(0)).flatten(2).sum(-1).cpu() / math.sqrt(n * probes)
        log(f"profile {min(offset + 4, n)}/{n}")
    sync(device)
    return linear.numpy(), signatures.flatten(2).numpy(), local_mse.numpy(), perf_counter() - start


def traverse(model, x, policies, device):
    """Exact shared-prefix evaluation for one image batch and selected policies."""
    result = {}
    counts = np.zeros(4, dtype=int)
    with torch.no_grad():
        root = model.stem(x.to(device))
        def visit(h, depth, ids):
            if depth == 4:
                logits = model.head(h).cpu()
                for i in ids:
                    result[i] = logits
                return
            for mode in sorted(set(int(policies[i, depth]) for i in ids)):
                group = [i for i in ids if policies[i, depth] == mode]
                counts[depth] += len(x)
                visit(model.stages[depth](h, mode), depth + 1, group)
        visit(root, 0, list(range(len(policies))))
    sync(device)
    return torch.stack([result[i] for i in range(len(policies))]), counts


def exhaustive(model, x, teacher, y, policies, device):
    values = np.zeros((len(policies), len(x), 3), dtype=np.float32)
    counts = np.zeros(4, dtype=int)
    start = perf_counter()
    for offset in range(0, len(x), BATCH_SIZE):
        batch_logits, used = traverse(model, x[offset:offset + BATCH_SIZE], policies, device)
        for i, logits in enumerate(batch_logits):
            values[i, offset:offset + len(logits)] = losses(logits, teacher[offset:offset + len(logits)], y[offset:offset + len(logits)]).numpy()
        counts += used
        log(f"exhaustive {min(offset + BATCH_SIZE, len(x))}/{len(x)} images; {len(policies)} policies")
    return values, counts, perf_counter() - start


def key(values):
    return (float(values[:, 2].sum()), float(values[:, 0].mean()))


def accuracy_race(model, x, teacher, y, policies, device, initial):
    """Exact empirical top-1 optimum; teacher KL breaks ties.

    A candidate is eliminated only if its already observed mistakes exceed
    a completely evaluated incumbent's total. Unseen mistakes cannot be negative.
    """
    start = perf_counter()
    order = torch.argsort(-teacher[torch.arange(len(y)), y].clamp_min(1e-30).log(), descending=True)
    ordered_x, ordered_teacher, ordered_y = x[order], teacher[order], y[order]
    initial_logits = batch_forward(model, ordered_x, policies[initial], device)
    incumbent = losses(initial_logits, ordered_teacher, ordered_y).numpy()
    bound = int(incumbent[:, 2].sum())
    values = np.zeros((len(policies), len(x), 3), dtype=np.float32)
    values[initial] = incumbent
    seen = np.zeros(len(policies), dtype=int)
    seen[initial] = len(x)
    active = [i for i in range(len(policies)) if i != initial]
    stage_counts = np.full(4, len(x), dtype=int)
    for offset in range(0, len(x), BATCH_SIZE):
        if not active:
            break
        batch_logits, used = traverse(model, ordered_x[offset:offset + BATCH_SIZE], policies[active], device)
        stage_counts += used
        survivors = []
        for local_id, policy_id in enumerate(active):
            batch = len(batch_logits[local_id])
            values[policy_id, offset:offset + batch] = losses(batch_logits[local_id], ordered_teacher[offset:offset + batch], ordered_y[offset:offset + batch]).numpy()
            seen[policy_id] += batch
            if values[policy_id, :, 2].sum() <= bound:
                survivors.append(policy_id)
        active = survivors
    complete = np.flatnonzero(seen == len(x))
    best = min(complete.tolist(), key=lambda i: key(values[i]))
    final_errors = int(values[best, :, 2].sum())
    eliminated = np.flatnonzero(seen < len(x))
    assert all(values[i, :, 2].sum() > final_errors for i in eliminated)
    sync(device)
    return best, {"seconds": perf_counter() - start, "initial_policy_id": int(initial), "initial_total_errors": bound, "winner_errors": final_errors, "winner_kl": key(values[best])[1], "policy_image_evaluations": int(seen.sum()), "naive_policy_image_evaluations": len(policies) * len(x), "stage_image_counts": stage_counts.tolist(), "fully_evaluated_policies": len(complete), "eliminated_policies": len(eliminated), "certificate": [{"policy": policies[i].tolist(), "images_seen": int(seen[i]), "mistakes_seen": int(values[i, :, 2].sum()), "complete": bool(seen[i] == len(x))} for i in range(len(policies))]}


def calibrated_model(device):
    original = resnet50(weights=None).eval()
    original.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    original.requires_grad_(False)
    model = QuantResNet(original).eval().to(device)
    manifest = read_manifest()
    for mod in model.modules():
        if isinstance(mod, Activation):
            mod.observe = True
    with torch.no_grad():
        for offset in range(0, len(manifest["observer"]), BATCH_SIZE):
            observer_x, _ = load_images(manifest["observer"][offset:offset + BATCH_SIZE])
            model(observer_x.to(device))
    sync(device)
    for mod in model.modules():
        if isinstance(mod, Activation):
            mod.observe = False
    return model, manifest


def compare():
    """Frozen methods, nine resource budgets, exact objectives and real timings."""
    torch.set_num_threads(2)
    device = select_device()
    model, manifest = calibrated_model(device)
    x, y = load_images(manifest["search"])
    teacher_logits = batch_forward(model, x, None, device)
    teacher = teacher_logits.softmax(-1)
    all_policies = np.array(list(itertools.product(range(4), repeat=4)))
    size, bitops, _, resource = costs(model, all_policies)
    # Fix the grid from resources alone; no accuracy-dependent budget adjustment.
    budgets = list(itertools.product((0.70, 0.85, 0.95), (0.60, 0.75, 0.90)))
    masks = [(size <= m * resource["all_w8a8_bytes"]) & (bitops <= c * resource["all_w8a8_bitops"]) for m, c in budgets]
    union = np.logical_or.reduce(masks)
    policies = all_policies[union]
    profiles = np.load(OUT / "profiles.npz")
    linear, signatures, local_mse = profiles["linear"], profiles["signatures"], profiles["local_mse"]
    chosen = signatures[np.arange(4)[None, :], policies]
    first = linear[np.arange(4)[None, :], policies].sum(-1)
    scores = {
        "coupled": first + 0.5 * np.square(chosen.sum(1)).sum(-1),
        "additive": first + 0.5 * np.square(chosen).sum((1, 2)),
        "local_mse": local_mse[np.arange(4)[None, :], policies].sum(-1),
    }
    log(f"Exhaustive union for nine budgets: {len(policies)} policies")
    oracle, oracle_counts, oracle_seconds = exhaustive(model, x, teacher, y, policies, device)
    np.savez(OUT / "grid_oracle.npz", policies=policies, per_image=oracle, bytes=size[union], bitops=bitops[union], **scores)
    grids = []
    selected = set()
    for (memory, compute), full_mask in zip(budgets, masks):
        ids = np.flatnonzero(full_mask[union])
        exact = min(ids.tolist(), key=lambda i: key(oracle[i]))
        methods = {name: int(ids[np.argmin(score[ids])]) for name, score in scores.items()}
        for name, score in scores.items():
            shortlist = ids[np.argsort(score[ids])[:4]]
            methods[f"{name}_top4"] = min(shortlist.tolist(), key=lambda i: key(oracle[i]))
        methods["exact"] = exact
        initial = methods["local_mse"]
        local_initial = int(np.flatnonzero(ids == initial)[0])
        best, racing = accuracy_race(model, x, teacher, y, policies[ids], device, local_initial)
        race_id = int(ids[best])
        assert race_id == exact, (memory, compute, race_id, exact)
        methods["certified_race"] = race_id
        for name, policy_id in methods.items():
            export_policy_csv(model, policies[policy_id], OUT / "policies" / f"m{memory:.2f}_c{compute:.2f}_{name}.csv")
        selected.update(methods.values())
        grid = {"memory_fraction": memory, "compute_fraction": compute, "feasible": len(ids), "exact_policy": policies[exact].tolist(), "exact_errors": key(oracle[exact])[0], "race": racing, "methods": {name: {"id": i, "policy": policies[i].tolist(), "search_errors": key(oracle[i])[0], "search_kl": key(oracle[i])[1], "error_regret": key(oracle[i])[0] - key(oracle[exact])[0]} for name, i in methods.items()}}
        grids.append(grid)
        log(f"budget M={memory} C={compute} feasible={len(ids)} exact={policies[exact].tolist()} race={racing['seconds']:.2f}s inspected={racing['policy_image_evaluations']}/{racing['naive_policy_image_evaluations']}")
        (OUT / "grid_progress.json").write_text(json.dumps(grids, indent=2) + "\n")
    test_x, test_y = load_images(manifest["test"])
    test_logits = batch_forward(model, test_x, None, device)
    test_teacher = test_logits.softmax(-1)
    np.savez(OUT / "test_float.npz", predicted=test_logits.argmax(-1).numpy(), labels=test_y.numpy())
    for policy_id in sorted(selected):
        policy = policies[policy_id]
        path = OUT / ("test_" + "_".join(map(str, policy)) + ".npz")
        logits = batch_forward(model, test_x, policy, device)
        measured = losses(logits, test_teacher, test_y).numpy()
        np.savez(path, losses=measured, predicted=logits.argmax(-1).numpy(), labels=test_y.numpy())
        for grid in grids:
            for row in grid["methods"].values():
                if row["id"] == policy_id:
                    row["test_accuracy"] = float(1 - measured[:, 2].mean())
                    row["test_kl"] = float(measured[:, 0].mean())
    # Timed direct enumeration and prefix sharing use identical policies/batches.
    main_mask = masks[4][union]
    main_policies = policies[main_mask]
    times = []
    for repetition in range(3):
        start = perf_counter()
        direct = []
        for policy in main_policies:
            logits = batch_forward(model, x, policy, device)
            direct.append(losses(logits, teacher, y).numpy())
        direct_seconds = perf_counter() - start
        shared, _, shared_seconds = exhaustive(model, x, teacher, y, main_policies, device)
        assert np.array_equal(np.array(direct)[:, :, 2], shared[:, :, 2])
        assert np.allclose(np.array(direct)[:, :, 0], shared[:, :, 0], atol=2e-5, rtol=2e-4)
        initial = int(np.argmin(scores["local_mse"][main_mask]))
        _, race_timing = accuracy_race(model, x, teacher, y, main_policies, device, initial)
        times.append({"direct_seconds": direct_seconds, "prefix_seconds": shared_seconds, "race_seconds": race_timing["seconds"]})
        log(f"timing repetition {repetition}: {times[-1]}")
    result = {"device": str(device), "grid": grids, "oracle_union_seconds": oracle_seconds, "oracle_union_stage_image_counts": oracle_counts.tolist(), "timing_repetitions": times, "timing_scope": "Search evaluation after common model construction/calibration. Race excludes local-MSE seed profiling; that cost must be added for an end-to-end comparison.", "float_test_accuracy": float(test_logits.argmax(-1).eq(test_y).float().mean())}
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    log("Completed comparison; resnet_results/comparison.json")


def local_mse_profile(model, x, device):
    sync(device)
    start = perf_counter()
    local_mse = torch.zeros(4, 4)
    with torch.no_grad():
        for part in x.split(4):
            _, inputs, outputs = model(part.to(device), (3, 3, 3, 3), retain=True)
            for i in range(4):
                for mode in range(3):
                    error = model.stages[i](inputs[i], mode) - outputs[i]
                    local_mse[i, mode] += error.square().flatten(1).mean(-1).sum().cpu() / len(x)
    sync(device)
    return local_mse.numpy(), perf_counter() - start


def search(memory_fraction, compute_fraction, evaluate=False):
    """Standalone automatic policy selection; does not run an exhaustive oracle."""
    torch.set_num_threads(2)
    device = select_device()
    total_start = perf_counter()
    model, manifest = calibrated_model(device)
    x, y = load_images(manifest["search"])
    teacher = batch_forward(model, x, None, device).softmax(-1)
    profile_x, _ = load_images(manifest["profile"])
    policies = np.array(list(itertools.product(range(4), repeat=4)))
    size, bitops, _, resource = costs(model, policies)
    mask = (size <= memory_fraction * resource["all_w8a8_bytes"]) & (bitops <= compute_fraction * resource["all_w8a8_bitops"])
    if not mask.any():
        raise ValueError("No feasible stage policy for these budgets.")
    policies, size, bitops = policies[mask], size[mask], bitops[mask]
    local_mse, seed_seconds = local_mse_profile(model, profile_x, device)
    initial = int(np.argmin(local_mse[np.arange(4)[None, :], policies].sum(-1)))
    winner, certificate = accuracy_race(model, x, teacher, y, policies, device, initial)
    result = {"policy": policies[winner].tolist(), "stage_weight_activation_bits": [MODES[m] for m in policies[winner]], "fixed_stem_head_bits": [8, 8], "packed_bytes": float(size[winner]), "bitops": float(bitops[winner]), "memory_fraction_budget": memory_fraction, "compute_fraction_budget": compute_fraction, "feasible_policies": len(policies), "seed_seconds": seed_seconds, "search_and_seed_seconds": seed_seconds + certificate["seconds"], "total_seconds_including_loading_calibration": perf_counter() - total_start, "device": str(device), "certificate": certificate}
    result["activation_bounds"] = {name: float(m.bound.cpu()) for name, m in model.named_modules() if isinstance(m, Activation)}
    result["checkpoint_sha256"] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    result["manifest"] = str(MANIFEST)
    result["batch_size"] = BATCH_SIZE
    result["torch_version"] = str(torch.__version__)
    result["quantization_rounding"] = "nearest_ties_to_even"
    result["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    (OUT / "selected_policy.json").write_text(json.dumps(result, indent=2) + "\n")
    export_policy_csv(model, policies[winner], OUT / "selected_policy.csv")
    log(json.dumps({k: v for k, v in result.items() if k not in ("certificate", "activation_bounds")}, indent=2))
    if evaluate:
        evaluate_test(model, manifest, policies[winner], device)


def evaluate_test(model, manifest, policy, device):
    """Stream validation images and per-image predictions; retain no full image/logit tensor."""
    rows = manifest.get("test", [])
    if not rows:
        raise ValueError("No test images provided in the manifest.")
    policy = validate_policy(policy)
    export_policy_csv(model, policy, OUT / "selected_policy.csv")
    totals = {"float_top1_correct": 0, "mixed_top1_correct": 0, "float_top5_correct": 0, "mixed_top5_correct": 0}
    kl_sum = 0.0
    sync(device)
    start = perf_counter()
    with (OUT / "evaluation_predictions.csv").open("w", newline="") as stream, torch.no_grad():
        writer = csv.writer(stream)
        writer.writerow(["file", "label", "float_prediction", "mixed_prediction", "float_correct", "mixed_correct"])
        for offset in range(0, len(rows), BATCH_SIZE):
            part = rows[offset:offset + BATCH_SIZE]
            x, y = load_images(part)
            x = x.to(device)
            reference, logits = model(x).cpu(), model(x, policy).cpu()
            rp, qp = reference.argmax(-1), logits.argmax(-1)
            totals["float_top1_correct"] += int(rp.eq(y).sum())
            totals["mixed_top1_correct"] += int(qp.eq(y).sum())
            totals["float_top5_correct"] += int(reference.topk(5).indices.eq(y[:, None]).any(-1).sum())
            totals["mixed_top5_correct"] += int(logits.topk(5).indices.eq(y[:, None]).any(-1).sum())
            kl_sum += float(losses(logits, reference.softmax(-1), y)[:, 0].sum())
            writer.writerows([r["file"], int(label), int(a), int(b), int(a == label), int(b == label)] for r, label, a, b in zip(part, y, rp, qp))
            if offset == 0 or (offset // BATCH_SIZE + 1) % 25 == 0 or offset + BATCH_SIZE >= len(rows):
                log(f"evaluate {min(offset + BATCH_SIZE, len(rows))}/{len(rows)}")
    sync(device)
    result = {"images": len(rows), "policy": list(policy), "device": str(device), "batch_size": BATCH_SIZE, "seconds": perf_counter() - start, "teacher_kl": kl_sum / len(rows), **totals}
    result.update({key.replace("_correct", "_accuracy"): value / len(rows) for key, value in totals.items()})
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    log(json.dumps(result, indent=2))
    return result


def evaluate_saved(policy_path, export_only=False):
    saved = json.loads(policy_path.read_text())
    if saved.get("granularity") == "operator":
        import sys
        from .fisher_router_v2 import evaluate_saved as evaluate_layer_policy
        return evaluate_layer_policy(sys.modules[__name__], saved, export_only)
    if "quantization_rounding" not in saved:
        log("Policy predates explicit ties-to-even rounding. Re-evaluation uses corrected rounding; original MPS results may differ.")
    policy = validate_policy(saved["policy"])
    if saved.get("checkpoint_sha256") and saved["checkpoint_sha256"] != hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest():
        raise ValueError("Saved policy checkpoint hash does not match --checkpoint.")
    device = select_device()
    model, manifest = calibrated_model(device)
    if "activation_bounds" in saved:
        modules = {name: m for name, m in model.named_modules() if isinstance(m, Activation)}
        if modules.keys() != saved["activation_bounds"].keys():
            raise ValueError("Saved activation quantizers do not match the model.")
        for name, value in saved["activation_bounds"].items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid calibration bound for {name}.")
            modules[name].bound.fill_(value)
    else:
        log("Legacy policy has no saved scales; recalibrating from the observer split.")
    export_policy_csv(model, policy, OUT / "selected_policy.csv")
    if not export_only:
        evaluate_test(model, manifest, policy, device)


def pareto_indices(storage, errors):
    """Minimize storage and mistakes; preserve ties at identical coordinates."""
    storage, errors = np.asarray(storage), np.asarray(errors)
    return [i for i in range(len(storage)) if not np.any(
        (storage <= storage[i]) & (errors <= errors[i]) &
        ((storage < storage[i]) | (errors < errors[i])))]


def frontier_candidates(policies, storage, bitops, measured):
    """One measured search winner per storage level, then nondominated levels."""
    winners = []
    for size in sorted(set(storage)):
        ids = np.flatnonzero(storage == size)
        winners.append(min(ids, key=lambda i: (*key(measured[i]), bitops[i], int(i))))
    front = pareto_indices(storage[winners], [key(measured[i])[0] for i in winners])
    return [int(winners[i]) for i in front]


def plot_frontier(rows, path, title, note, flag="validation_frontier", x_metric="compression_ratio"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scale = 1e9 if x_metric == "bitops" else 1
    def x(row):
        return row[x_metric] / scale
    floating = rows[0]
    quantized = rows[1:]
    ax.scatter([x(r) for r in quantized], [r["top1_percent"] for r in quantized],
               color="#718096", s=55, label="Evaluated quantized policies", zorder=3)
    front = sorted((r for r in rows if r["id"] != "FP32" and r[flag]), key=x)
    ax.plot([x(r) for r in front], [r["top1_percent"] for r in front],
            "o-", color="#137c66", linewidth=2, label="Nondominated evaluated points", zorder=4)
    ax.scatter([x(floating)], [floating["top1_percent"]], color="#263248", marker="*", s=160,
               label=f"FP32 baseline ({floating['top1_percent']:.2f}%)", zorder=5)
    for i, row in enumerate(rows):
        ax.annotate(row["id"], (x(row), row["top1_percent"]),
                    xytext=(5, 9 if i % 2 else -15), textcoords="offset points", fontsize=9)
    ax.axhline(floating["top1_percent"], color="#263248", linestyle=":", alpha=.35)
    ax.set_xlabel("Estimated compute (GigaBitOps; lower is less compute)" if x_metric == "bitops" else
                  "Compression ratio vs FP32 (estimated model storage; higher is smaller)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(title, loc="left", fontsize=13, pad=14)
    ax.grid(alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=.08, y=.20)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(.08, .025, note, fontsize=8, color="#4a5568")
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(path, dpi=180)
    fig.savefig(Path(path).with_suffix(".svg"))
    plt.close(fig)


def frontier():
    """Search all stage policies; validate only the search frontier and two anchors."""
    # Fail before expensive inference if the plotting dependency is missing.
    import matplotlib
    torch.set_num_threads(2)
    device = select_device()
    model, manifest = calibrated_model(device)
    if not manifest.get("test"):
        raise ValueError("A held-out test split is required to plot validated accuracy.")
    x, y = load_images(manifest["search"])
    teacher_logits = batch_forward(model, x, None, device)
    teacher = teacher_logits.softmax(-1)
    policies = np.array(list(itertools.product(range(4), repeat=4)))
    size, bitops, _, resource = costs(model, policies)
    floating_bytes = sum(4 * (m.count + (m.op.bias.numel() if m.op.bias is not None else 0))
                         for m in model.modules() if isinstance(m, Affine))
    log("Searching all 256 stage policies without size/compute budgets.")
    measured, counts, seconds = exhaustive(model, x, teacher, y, policies, device)
    np.savez(OUT / "frontier_search.npz", policies=policies, bytes=size, bitops=bitops, per_image=measured)
    front = frontier_candidates(policies, size, bitops, measured)
    anchors = [int(np.flatnonzero((policies == mode).all(1))[0]) for mode in (0, 2)]
    selected = sorted(set(front + anchors), key=lambda i: (size[i], key(measured[i]), bitops[i]))
    candidate_policies = policies[selected]
    log(f"Search frontier: {len(front)} policies; validating {len(selected)} including all-stage W4A8 and W8A8 references.")
    del x, teacher, teacher_logits
    # One floating reference and one shared-prefix traversal per image batch.
    test = manifest["test"]
    correct = np.zeros(len(selected) + 1, dtype=np.int64)
    sync(device)
    start = perf_counter()
    with (OUT / "frontier_predictions.csv").open("w", newline="") as stream, torch.no_grad():
        writer = csv.writer(stream)
        writer.writerow(["file", "label", "FP32"] + [f"P{i + 1:02}" for i in range(len(selected))])
        for offset in range(0, len(test), BATCH_SIZE):
            part = test[offset:offset + BATCH_SIZE]
            images, labels = load_images(part)
            reference = model(images.to(device)).cpu().argmax(-1)
            logits, _ = traverse(model, images, candidate_policies, device)
            predictions = torch.cat((reference[None], logits.argmax(-1)))
            correct += predictions.eq(labels[None]).sum(-1).numpy()
            writer.writerows([r["file"], int(labels[j]), *predictions[:, j].tolist()] for j, r in enumerate(part))
            if offset == 0 or (offset // BATCH_SIZE + 1) % 25 == 0 or offset + BATCH_SIZE >= len(test):
                log(f"frontier validation {min(offset + BATCH_SIZE, len(test))}/{len(test)}")
    sync(device)
    evaluation_seconds = perf_counter() - start
    accuracy = 100 * correct / len(test)
    validation_front = {i + 1 for i in pareto_indices(size[selected], -correct[1:])}
    rows = [{"id": "FP32", "policy": "float", "storage_bytes": floating_bytes,
             "compression_ratio": 1.0, "top1_percent": float(accuracy[0]), "correct": int(correct[0]),
             "images": len(test), "bitops_vs_w8a8": "", "search_frontier": False,
             "validation_frontier": False, "stage_modes": "FP32"}]
    bounds = {name: float(m.bound.cpu()) for name, m in model.named_modules() if isinstance(m, Activation)}
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    for j, policy_id in enumerate(selected, 1):
        policy = policies[policy_id]
        label = f"P{j:02}"
        rows.append({"id": label, "policy": ",".join(map(str, policy)), "storage_bytes": int(size[policy_id]),
                     "compression_ratio": float(floating_bytes / size[policy_id]), "top1_percent": float(accuracy[j]),
                     "correct": int(correct[j]), "images": len(test),
                     "bitops_vs_w8a8": float(bitops[policy_id] / resource["all_w8a8_bitops"]),
                     "search_frontier": policy_id in front, "validation_frontier": j in validation_front,
                     "stage_modes": " / ".join(f"W{MODES[m][0]}A{MODES[m][1]}" for m in policy)})
        export_policy_csv(model, policy, OUT / "policies" / f"{label}.csv")
        saved = {"policy": policy.tolist(), "activation_bounds": bounds, "checkpoint_sha256": digest,
                 "manifest": str(MANIFEST), "packed_bytes": int(size[policy_id]), "bitops": float(bitops[policy_id]),
                 "quantization_rounding": "nearest_ties_to_even"}
        (OUT / "policies" / f"{label}.json").write_text(json.dumps(saved, indent=2) + "\n")
    with (OUT / "frontier.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"device": str(device), "torch_version": str(torch.__version__), "search_images": len(y),
               "quantization_rounding": "nearest_ties_to_even", "batch_size": BATCH_SIZE,
               "validation_images": len(test), "validation_classes": len(set(r["label"] for r in test)),
               "search_space_size": 256, "search_frontier_size": len(front), "validated_policies": len(selected),
               "search_seconds": seconds, "validation_seconds": evaluation_seconds,
               "stage_image_counts": counts.tolist(), "floating_storage_bytes": floating_bytes,
               "compression_definition": "FP32 folded weights+biases / quantized packed weights+scales+biases; activation working memory excluded.",
               "scope": "Exact search-set frontier among 256 stage policies. Validation frontier is only among selected candidates and references; not a global ImageNet frontier.",
               "candidates": rows}
    (OUT / "frontier.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_frontier(rows, OUT / "frontier.png",
                  f"ResNet-50: compression vs held-out top-1 accuracy ({len(test):,} images)",
                  f"{summary['validation_classes']} classes; candidates selected on {len(y)} separate search images.\n"
                  "Estimated storage; fake quantization. Line connects measured candidates, not interpolated models.")
    log(json.dumps(summary, indent=2))


def seed_timing():
    """Measure standalone local-MSE profiling, including anchor forwards."""
    torch.set_num_threads(2)
    device = select_device()
    model, manifest = calibrated_model(device)
    x, _ = load_images(manifest["profile"])
    expected = np.load(OUT / "profiles.npz")["local_mse"]
    times = []
    for repetition in range(3):
        local_mse, seconds = local_mse_profile(model, x, device)
        times.append(seconds)
        assert np.allclose(local_mse, expected, atol=2e-5, rtol=2e-4)
        log(f"Standalone MSE profiling repetition {repetition}: {times[-1]:.4f}s")
    (OUT / "seed_timing.json").write_text(json.dumps({"device": str(device), "seconds": times, "scope": "Standalone local MSE profiling on 50 images, including anchor forward passes; excludes common calibration/loading."}, indent=2) + "\n")


def run():
    OUT.mkdir(exist_ok=True)
    device = select_device()
    torch.set_num_threads(2)
    torch.manual_seed(20261111)
    log(f"device={device}")
    manifest = read_manifest()
    original = resnet50(weights=None).eval()
    original.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    original.requires_grad_(False)
    model = QuantResNet(original).eval()
    observer_x, _ = load_images(manifest["observer"])
    with torch.no_grad():
        reference = original(observer_x[:2])
        folded = model(observer_x[:2])
        mismatch = float((reference - folded).abs().max())
        assert mismatch < 2e-4, mismatch
    del original
    model.to(device)
    start = perf_counter()
    for mod in model.modules():
        if isinstance(mod, Activation):
            mod.observe = True
    batch_forward(model, observer_x, None, device)
    for mod in model.modules():
        if isinstance(mod, Activation):
            mod.observe = False
    calibration_seconds = perf_counter() - start
    del observer_x
    profile_x, profile_y = load_images(manifest["profile"])
    search_x, search_y = load_images(manifest["search"])
    profile_teacher = batch_forward(model, profile_x, None, device).softmax(-1)
    search_teacher_logits = batch_forward(model, search_x, None, device)
    search_teacher = search_teacher_logits.softmax(-1)
    policy_space = np.array(list(itertools.product(range(4), repeat=4)), dtype=np.int64)
    size, bitops, mask, resource = costs(model, policy_space)
    feasible = policy_space[mask]
    log(f"feasible={len(feasible)}/256; resource={resource}")
    linear, signatures, local_mse, profiling_seconds = profile(model, profile_x, profile_teacher, device)
    np.savez(OUT / "profiles.npz", linear=linear, signatures=signatures, local_mse=local_mse, policies=feasible)
    chosen = signatures[np.arange(4)[None, :], feasible]
    first = linear[np.arange(4)[None, :], feasible].sum(-1)
    score = first + 0.5 * np.square(chosen.sum(1)).sum(-1)
    additive = first + 0.5 * np.square(chosen).sum((1, 2))
    mse = local_mse[np.arange(4)[None, :], feasible].sum(-1)
    oracle, counts, oracle_seconds = exhaustive(model, search_x, search_teacher, search_y, feasible, device)
    np.savez(OUT / "oracle.npz", policies=feasible, per_image=oracle, bytes=size[mask], bitops=bitops[mask], coupled=score, additive=additive, mse=mse)
    best_accuracy = min(range(len(feasible)), key=lambda i: key(oracle[i]))
    best_kl = int(np.argmin(oracle[:, :, 0].mean(-1)))
    rng = np.random.default_rng(20261111)
    methods = {
        "coupled_direct": [int(np.argmin(score))],
        "additive_direct": [int(np.argmin(additive))],
        "local_mse_direct": [int(np.argmin(mse))],
        "coupled_shortlist": np.argsort(score)[:min(8, len(feasible))].tolist(),
        "additive_shortlist": np.argsort(additive)[:min(8, len(feasible))].tolist(),
        "random_shortlist": rng.choice(len(feasible), min(8, len(feasible)), replace=False).tolist(),
        "exact_accuracy_oracle": [best_accuracy],
        "exact_kl_oracle": [best_kl],
    }
    selection = {name: min(ids, key=lambda i: key(oracle[i])) for name, ids in methods.items()}
    for name, policy_id in selection.items():
        export_policy_csv(model, feasible[policy_id], OUT / "policies" / f"{name}.csv")
    export_policy_csv(model, feasible[best_accuracy], OUT / "selected_policy.csv")
    selected_policies = {tuple(feasible[i]) for i in selection.values()} | {(0, 0, 0, 0), (2, 2, 2, 2), (3, 3, 3, 3)}
    test_x, test_y = load_images(manifest["test"])
    test_teacher_logits = batch_forward(model, test_x, None, device)
    test_teacher = test_teacher_logits.softmax(-1)
    test_metrics = {}
    for policy in sorted(selected_policies):
        start = perf_counter()
        logits = batch_forward(model, test_x, policy, device)
        measured = losses(logits, test_teacher, test_y).numpy()
        label = ",".join(map(str, policy))
        test_metrics[label] = {"accuracy": float(1 - measured[:, 2].mean()), "teacher_kl": float(measured[:, 0].mean()), "ce": float(measured[:, 1].mean()), "seconds": perf_counter() - start}
        np.savez(OUT / f"test_{label.replace(',', '_')}.npz", losses=measured, predicted=logits.argmax(-1).numpy(), labels=test_y.numpy())
        log(f"test {policy}: {test_metrics[label]}")
    summary = {
        "device": str(device), "torch": torch.__version__, "floating_fold_max_logit_error": mismatch,
        "search_space": "Four uniform stage groups; stem/classifier W8A8. No claim of per-layer global optimality.",
        "feasible_policies": len(feasible), "resource": resource,
        "quantizer": "BN folded; per-output-channel symmetric max-abs weights; per-tensor static symmetric signed activations; floating biases/accumulators/addition; no QAT/reconstruction.",
        "calibration_seconds": calibration_seconds, "profiling_seconds": profiling_seconds, "oracle_seconds_shared_prefix": oracle_seconds, "oracle_stage_image_counts": counts.tolist(),
        "float_search_accuracy": float(search_teacher_logits.argmax(-1).eq(search_y).float().mean()),
        "float_test_accuracy": float(test_teacher_logits.argmax(-1).eq(test_y).float().mean()),
        "rank_correlation_coupled_kl": float(spearmanr(score, oracle[:, :, 0].mean(-1)).statistic),
        "rank_correlation_additive_kl": float(spearmanr(additive, oracle[:, :, 0].mean(-1)).statistic),
        "selection": {name: {"policy_id": int(i), "policy": feasible[i].tolist(), "search_errors": key(oracle[i])[0], "search_kl": key(oracle[i])[1], "candidate_evaluations": len(methods[name]), "test": test_metrics[",".join(map(str, feasible[i]))]} for name, i in selection.items()},
        "test_policy_metrics": test_metrics,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(json.dumps(summary, indent=2))


def main(argv=None):
    global DEVICE, BATCH_SIZE, PROFILE_BATCH_SIZE, CHECKPOINT, OUT, MANIFEST, IMAGE_ROOT
    parser = RunArgumentParser()
    parser.add_argument("phase", choices=["prepare", "run", "compare", "seed-timing", "search", "evaluate", "export", "frontier", "fisher-frontier"])
    parser.add_argument("--memory-fraction", type=float, help="Budgeted search only; default 0.85.")
    parser.add_argument("--compute-fraction", type=float, help="BitOps cap relative to all-W8A8: fisher-frontier has no cap unless explicitly set; legacy search defaults to 0.75.")
    parser.add_argument("--device", default="auto", help="auto prefers CUDA, then MPS, then CPU; or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference/search and observer calibration batch size.")
    parser.add_argument("--profile-batch-size", type=int, default=4, help="Fisher-frontier backward microbatch size; lower independently if GPU memory is tight.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="JSON splits with file paths and original ImageNet labels 0..999.")
    parser.add_argument("--image-root", type=Path, help="Base for relative manifest image paths (default: custom manifest's directory).")
    parser.add_argument("--imagenet-root", type=Path, help="Directory containing train/<WNID> and val/<WNID> folders.")
    parser.add_argument("--observer-images", type=int, default=1024)
    parser.add_argument("--profile-images", type=int, default=128)
    parser.add_argument("--search-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--evaluate", action="store_true", help="After search, evaluate both float and mixed models on all test images.")
    parser.add_argument("--policy", type=Path, help="Saved selected_policy.json for evaluate/export.")
    parser.add_argument("--router-mode", choices=["transfer", "local", "ilp"], default="transfer", help="V2 Fisher frontier: transfer=group repair, local=one/two-layer refinement, ilp=seeds only. Edit MODES in amp/fisher_router_v2.py to choose formats.")
    parser.add_argument("--budget-points", type=int, default=8, help="Fisher frontier storage budgets (at least 2).")
    parser.add_argument("--max-candidates", type=int, default=64, help="Fisher frontier unique evaluation allowance, including seeds and early rejections.")
    parser.add_argument("--profile-cache", type=Path, help="Reusable Fisher scores, costs and calibration; mismatching fingerprints are rejected.")
    args = parser.parse_args(argv)
    default_out = ROOT / "runs" / ("fisher_frontier_v2" if args.phase == "fisher-frontier" else "v2_" + args.phase)
    OUT = (args.output_dir or default_out).expanduser().resolve()
    # Check before FileHandler can append to an existing experiment's log.
    if OUT.exists() and (not OUT.is_dir() or any(OUT.iterdir())):
        parser.error(f"V2 requires a fresh output directory: {OUT}. Choose a new --output-dir; existing files were not changed.")
    from .fisher_router_v2 import validate_modes
    validate_modes()
    with run_logging(OUT, args.phase):
        command = (getattr(sys, "orig_argv", [sys.executable, *sys.argv]) if argv is None else
                   [sys.executable, str(ROOT / "resnet_benchmark_v2.py"), *argv])
        log("[run] Command: " + shlex.join(command))
        if args.phase == "frontier" and (args.memory_fraction is not None or args.compute_fraction is not None):
            parser.error("frontier explores size/accuracy trade-offs without memory/compute fractions; omit those flags.")
        if args.phase == "fisher-frontier" and args.memory_fraction is not None:
            parser.error("fisher-frontier generates storage budgets automatically; omit --memory-fraction.")
        if args.phase == "fisher-frontier" and (args.budget_points < 2 or args.max_candidates <= 0):
            parser.error("Fisher frontier requires at least two budgets and a positive candidate allowance.")
        args.memory_fraction = 0.85 if args.memory_fraction is None else args.memory_fraction
        if args.compute_fraction is None and args.phase != "fisher-frontier":
            args.compute_fraction = 0.75
        fractions = [args.memory_fraction] + ([] if args.compute_fraction is None else [args.compute_fraction])
        if args.batch_size <= 0 or args.profile_batch_size <= 0 or not all(math.isfinite(v) and v > 0 for v in fractions):
            parser.error("Batch size and budget fractions must be positive and finite.")
        if args.imagenet_root and args.manifest:
            parser.error("Choose --imagenet-root or --manifest, not both.")
        if (args.imagenet_root or args.manifest) and args.phase not in ("search", "evaluate", "export", "frontier", "fisher-frontier"):
            parser.error("Custom datasets support search/evaluate/export/frontier/fisher-frontier; run/compare are the original small pilot benchmark.")
        if args.evaluate and args.phase != "search":
            parser.error("--evaluate is for search; use the evaluate phase to evaluate a saved policy.")
        if args.policy and args.phase not in ("evaluate", "export"):
            parser.error("--policy is only used by evaluate/export.")
        DEVICE, BATCH_SIZE, PROFILE_BATCH_SIZE = args.device, args.batch_size, args.profile_batch_size
        log("[run] Configuration: " + json.dumps(vars(args), default=str, sort_keys=True))
        CHECKPOINT = args.checkpoint.expanduser().resolve()
        if not CHECKPOINT.is_file():
            parser.error(f"Checkpoint not found: {CHECKPOINT}. No assets are downloaded automatically.")
        if args.phase != "prepare":
            log(f"[device] Selecting {args.device}; checking runtime before dataset preparation.")
            selected_device = select_device()
            runtime = device_runtime(selected_device, check=True)
            (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
            log("[device] Runtime: " + json.dumps(runtime, sort_keys=True))
            log(f"device={selected_device}, batch_size={BATCH_SIZE}, output={OUT}")
            if selected_device.type == "cuda":
                log(f"CUDA preflight passed: {runtime['gpu_name']}, Torch CUDA {runtime['torch_cuda_version']}.")
        if args.manifest:
            MANIFEST = args.manifest.expanduser().resolve()
            IMAGE_ROOT = (args.image_root or MANIFEST.parent).expanduser().resolve()
        elif args.imagenet_root:
            MANIFEST = OUT / "dataset_manifest.json"
            generated = imagenet_manifest(args.imagenet_root, (args.observer_images, args.profile_images, args.search_images), args.seed)
            MANIFEST.write_text(json.dumps(generated, indent=2) + "\n")
            log(f"ImageNet split sizes: { {k: len(v) for k, v in generated.items()} }")
        elif args.image_root:
            IMAGE_ROOT = args.image_root.expanduser().resolve()
        torch.manual_seed(args.seed)
        policy_path = (args.policy or OUT / "selected_policy.json").expanduser().resolve()
        def fisher_frontier():
            import sys
            from .fisher_router_v2 import frontier as layer_frontier
            try:
                return layer_frontier(sys.modules[__name__], args)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("CUDA out of memory during Fisher frontier. Reduce --profile-batch-size (try 1) for backward profiling, or --batch-size for calibration/search/validation. Use a fresh profile-cache path if changing profiling settings. No CPU fallback was attempted.") from exc
        {"prepare": prepare, "run": run, "compare": compare, "seed-timing": seed_timing,
         "search": lambda: search(args.memory_fraction, args.compute_fraction, args.evaluate),
         "frontier": frontier,
         "fisher-frontier": fisher_frontier,
         "evaluate": lambda: evaluate_saved(policy_path),
         "export": lambda: evaluate_saved(policy_path, export_only=True)}[args.phase]()


if __name__ == "__main__":
    main()
