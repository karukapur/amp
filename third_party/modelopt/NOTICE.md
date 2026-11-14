The elementwise Fisher scoring functional in `amp/fisher_router.py` is adapted from
NVIDIA Model Optimizer, `modelopt/torch/quantization/algorithms.py`, function
`_get_auto_quantize_score`.

Source revision: 814e6b49ea6752498b5443d49bc8d39a3ca6ad65

https://github.com/NVIDIA/Model-Optimizer/blob/814e6b49ea6752498b5443d49bc8d39a3ca6ad65/modelopt/torch/quantization/algorithms.py

Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under the Apache License, Version 2.0; see LICENSE in this directory.

Local changes: sum labeled cross-entropy per batch, normalize accumulated scores
by image count, replay joint integer weight/activation alternatives at backward
hooks, snapshot outputs before in-place residual additions, and retain unprojected scores. This project does not vendor NVIDIA's
searcher or Shapley implementation. The router and cost accounting are local.
