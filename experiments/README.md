# Independent prototype

`transport_sketch_demo.py` is the earlier synthetic residual-MLP experiment. It
is separate from the ResNet Fisher/router method, but retained for reproducibility.
Run it from the repository root:

```sh
python -m experiments.transport_sketch_demo
```

It needs Torch, NumPy and SciPy and writes generated results to the ignored
`runs/transport_sketch/results.json`. The previous generated result is archived
locally in `artifacts/experiments/`.
