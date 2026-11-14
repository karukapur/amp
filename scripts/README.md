# Historical audit tools

Run these from the repository root so package imports resolve:

```sh
python -m scripts.audit_pareto
python -m scripts.audit_backend --help
python -m scripts.analyze_resnet_results
```

`audit_pareto` independently checks the retained stage-based pilot, including
data integrity, packing, statistics and CPU inference. Its input is the local
`artifacts/runs/pareto_pilot/` archive and it writes to
`artifacts/runs/pareto_audit/`. It also needs the checkpoint and retained pilot
images/manifest. It is not the Fisher router's search implementation.

`audit_backend` compares the historical CPU and Apple MPS arithmetic; it requires
an available MPS device and the audit artifacts. `--legacy-rounding` reproduces
the original arithmetic explicitly.

`analyze_resnet_results` reads the local legacy `resnet_results/` files and writes
its certificate/statistics audit there. Raw historical inputs are not shipped in
the cleaned source repository; selected conclusions remain under `reports/`.
