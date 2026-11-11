"""Trace device dependence with the frozen P04 bounds; no policy selection."""
import copy
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet50

import audit_pareto as audit
import resnet_benchmark as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-rounding", action="store_true", help="Reproduce the original device-native rounding.")
    parser.add_argument("--output-dir", type=Path, default=audit.OUT / "backend_fixed")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.legacy_rounding:
        runner.round_to_even = lambda x: x.round()
    torch.set_num_threads(2)
    device = runner.select_device("mps")
    saved = json.loads((audit.SOURCE / "policies/P04.json").read_text())
    original = resnet50(weights=None).eval().requires_grad_(False)
    original.load_state_dict(torch.load(runner.CHECKPOINT, weights_only=True))
    cpu = runner.QuantResNet(original).eval()
    for name, module in cpu.named_modules():
        if isinstance(module, runner.Activation):
            module.bound.fill_(saved["activation_bounds"][name])
    gpu = copy.deepcopy(cpu).to(device)
    manifest = runner.read_manifest()
    logits = []
    with torch.inference_mode():
        for offset in range(0, 500, 16):
            x, _ = runner.load_images(manifest["test"][offset:offset+16])
            logits.append(gpu(x.to(device), saved["policy"]).cpu().numpy())
    logits = np.concatenate(logits)
    np.savez(args.output_dir / "mps_P04_logits.npz", logits=logits)
    reference = np.load(audit.OUT / "cpu_P04_logits.npz")["logits"]
    labels = np.array([r["label"] for r in manifest["test"]])
    old = list(csv.DictReader((audit.SOURCE / "frontier_predictions.csv").open()))
    matches_saved = np.array_equal(logits.argmax(1), [int(r["P04"]) for r in old])
    if args.legacy_rounding:
        assert matches_saved
    differences = np.flatnonzero(reference.argmax(1) != logits.argmax(1))
    result = {"cpu_correct": int(sum(reference.argmax(1) == labels)), "mps_correct": int(sum(logits.argmax(1) == labels)),
        "legacy_rounding": args.legacy_rounding,
        "mps_direct_matches_saved_prefix_predictions": bool(matches_saved),
        "max_logit_difference": float(np.max(np.abs(reference-logits))),
        "mean_per_image_max_logit_difference": float(np.max(np.abs(reference-logits), axis=1).mean()),
        "disagreeing_images": [{"index": int(i), "file": manifest["test"][i]["file"], "label": int(labels[i]),
              "cpu_prediction": int(reference[i].argmax()), "mps_prediction": int(logits[i].argmax()),
              "cpu_top_two_margin": float(np.sort(reference[i])[-1]-np.sort(reference[i])[-2]),
              "mps_top_two_margin": float(np.sort(logits[i])[-1]-np.sort(logits[i])[-2])} for i in differences]}
    # Compare every activation quantizer and affine output on a mismatching image.
    i = int(differences[0]) if len(differences) else 0
    x, _ = runner.load_images([manifest["test"][i]])
    cpu_values, gpu_values, cpu_inputs = {}, {}, {}
    for name, module in cpu.named_modules():
        if isinstance(module, (runner.Activation, runner.Affine)):
            def hook(module, args, output, name=name):
                cpu_values[name] = output.detach().clone()
                if isinstance(module, runner.Activation):
                    cpu_inputs[name] = (args[0].detach().clone(), args[1])
            module.register_forward_hook(hook)
    for name, module in gpu.named_modules():
        if isinstance(module, (runner.Activation, runner.Affine)):
            module.register_forward_hook(lambda mod, args, output, name=name: gpu_values.__setitem__(name, output.detach().cpu().clone()))
    with torch.inference_mode():
        c = cpu(x, saved["policy"])
        g = gpu(x.to(device), saved["policy"]).cpu()
        result["trace_single_image_predictions"] = {"cpu": int(c.argmax()), "mps": int(g.argmax())}
        result["trace_max_errors"] = {name: float((cpu_values[name]-gpu_values[name]).abs().max()) for name in cpu_values}
        isolated = {}
        modules = dict(gpu.named_modules())
        for name, (input_tensor, bits) in cpu_inputs.items():
            # Same input and same saved bound: isolates quantizer arithmetic from convolution.
            output = modules[name](input_tensor.to(device), bits).cpu()
            scale = saved["activation_bounds"][name] / (2**(bits-1)-1)
            left = torch.round(cpu_values[name]/scale)
            right = torch.round(output/scale)
            isolated[name] = {"different_integer_codes": int((left != right).sum()), "elements": input_tensor.numel(),
                              "max_output_difference": float((cpu_values[name]-output).abs().max())}
        result["isolated_quantizers_same_input"] = isolated
        ties = torch.arange(-3.5, 4, 1)
        result["half_integer_rounding"] = {"inputs": ties.tolist(), "cpu": ties.round().tolist(), "mps": ties.to(device).round().cpu().tolist()}
        result["explicit_half_integer_rounding"] = runner.round_to_even(ties.to(device)).cpu().tolist()
        if not args.legacy_rounding:
            test_ties = torch.arange(-32767.5, 32768, 1)
            values = torch.cat((test_ties, torch.nextafter(test_ties, torch.full_like(test_ties, -float("inf"))),
                                torch.nextafter(test_ties, torch.full_like(test_ties, float("inf")))))
            assert torch.equal(runner.round_to_even(values.to(device)).cpu(), values.round())
            result["mps_rounding_regression_values_checked"] = len(values)
    (args.output_dir / "backend_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("trace_max_errors", "isolated_quantizers_same_input")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
