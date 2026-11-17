# Fisher + MILP versus NVIDIA AutoQuantize

The repository's Fisher + MILP baseline follows the same core algorithm as
NVIDIA AutoQuantize's gradient-based method. The scoring formula is explicitly
adapted from NVIDIA, as recorded in the [attribution notice](../third_party/modelopt/NOTICE.md).

The NVIDIA column below follows its [documented AutoQuantize implementation](https://nvidia.github.io/Model-Optimizer/announcements/20260824-autoquantize.html).
The local column describes the Fisher scoring and seed allocation shared by
the original implementation and v2, before any refinement or group repair.

| Aspect | NVIDIA AutoQuantize | Our Fisher + MILP baseline |
|---|---|---|
| **Sensitivity formula** | Squared output error weighted by squared gradients: sum of `(gradient × output error)²`. | Same core formula, with numerical clamping and image-count normalization. |
| **Profiling procedure** | One backward pass per scoring batch; locally replay operators for candidate formats. | Same approach, implemented with PyTorch hooks and explicit candidate replays. |
| **Loss function** | User supplies the task loss. | Fixed labeled cross-entropy, summed per microbatch. |
| **Allocation objective** | Minimize summed sensitivities under a resource budget. | Same additive objective. Neither baseline directly optimizes measured top-1 accuracy. |
| **Integer optimization** | Binary ILP: select one format per decision. | Independently implemented using SciPy's MILP solver, with a 10-second limit and feasibility checks. |
| **Resource model** | Parameter-weighted effective bits, including declared format overhead. | Estimated packed weight bytes, scales and biases; optional `MACs × W bits × A bits` cap. |
| **Decision granularity** | Supports deployment-driven operator groups, such as coupled projections. | Independently assigns formats to 54 ResNet operators; no equivalent deployment-coupling constraints. |
| **Quantization and deployment** | ModelOpt format configurations and supported export/runtime workflows. | Custom symmetric integer fake quantization in FP32; residual additions remain floating point. |

Calling the allocator “MILP” instead of “ILP” does not establish an algorithmic
difference here: both use binary format-selection variables.

The accurate characterization is **an adaptation of NVIDIA-style gradient
scoring with a separately implemented allocator and different quantization and
resource accounting**. Those differences alone do not establish methodological
novelty or a performance advantage. This comparison does not establish numerical
equivalence between the implementations.

V2's additional experimental mechanism is the **measured group-repair search
after MILP**, described in [the v2 usage document](group-repair-v2.md). Its
accuracy benefit remains unverified on full ImageNet.
