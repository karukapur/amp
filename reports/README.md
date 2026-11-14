# Retained evidence

- `fisher-pilot/`: the matched 24-candidate, capped local-versus-transfer pilot.
  It did not demonstrate a router frontier or evaluation-work advantage.
- `pareto-audit/`: the historical stage-policy audit and rounding correction.
  The original apparent accuracy gain is not an established improvement.
- `legacy-pilot/`: the earlier surrogate/enumeration report and exported policy.

These are historical records, not benchmarks for every later code revision.
Selected reports, comparison data and plots are kept with the source. Raw caches,
predictions, packed payloads, logs and development smoke runs were removed from
the versioned source tree and retained locally under `artifacts/runs/`. The
original `resnet_results/` inputs remain local and ignored because legacy pilot
commands and analysis use them. New runs write to the ignored `runs/` directory.

The HTML pilot replay embeds its own recorded data and remains usable offline
without the raw archive. Audit tools need their historical local inputs; see
`scripts/README.md`. No checkpoints or dataset images were removed by cleanup.
