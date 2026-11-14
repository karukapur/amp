**Novelty audit: mixed-precision frontier search with shared evaluation**

12 November 2026. Scope: the latest proposal in this conversation, assessed against primary papers and the existing local runner. This is a targeted prior-art audit, not an exhaustive systematic review or a guarantee about publication acceptance.

**Decision: publishable methodological novelty is not established.**

The current proposal is a potentially useful integration of known techniques. Its broad claim—choose related precision policies, reuse their computation, and allocate evaluation effort according to expected improvement and execution cost—is substantially anticipated by existing work. It should not currently be presented as a new research method.

This does not establish that an identical joint weight/activation implementation already exists, or that no systems contribution could emerge. It establishes that the present description does not identify a defensible technical advance beyond the closest methods.

**What was assessed**

The required pipeline retains sensitivity profiling and produces measured accuracy versus weight-storage compression, with a separate activation/compute constraint. The proposed middle step consists of reversible precision edits, batches with common prefixes, reuse of unchanged execution and calibration state, progressively larger screening sets, and prioritization by potential frontier improvement per incremental execution cost.

The confidential implementation was not accessed. Its baseline behavior is taken only from the user's description. The local ResNet runner is a separate source of evidence about work already implemented in this workspace.

**Closest prior art and the claims it limits**

| Proposed ingredient | Primary evidence | Consequence |
|---|---|---|
| Reversible compression exchanges and staged candidate evaluation | [EvoPress, ICML 2025](https://proceedings.mlr.press/v267/sieberling25a.html), Section 3, Algorithm 1 and design considerations. Its level-switch mutation increases compression in one unit and decreases it in another while preserving the constraint. Selection uses progressively larger token samples. | These search operations are established, including application to quantization. Joint W/A constraints would require an extension, but do not by themselves establish a new search principle. |
| Design candidate batches to share prefixes, execute a merged graph, and manage cached intermediates | [Li et al., Exploiting Reuse in Pipeline-Aware Hyperparameter Tuning, 2019 preprint](https://arxiv.org/abs/1903.05176), Sections 3–5. It formalizes common-prefix elimination, deliberately constructs reusable candidate graphs, integrates early stopping, and formulates cache management. | The claim that candidate selection and execution must be designed together for reuse is already explicit. This is the strongest overlap with the latest proposal. |
| Prefer candidates with high expected improvement and low cost after prefix reuse | [EEIPU, 2024 preprint](https://arxiv.org/abs/2411.03731), Section 3, Algorithm 1. Candidate generation retains cached prefixes; the acquisition discounts memoized stages and models remaining evaluation cost. | Improvement per additional execution cost is already a concrete algorithmic idea, rather than merely a general inspiration. |
| Cheap quantized policy construction, a learned quality proxy, and iterative Pareto search | [AMQ, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1799/), Sections 3.3–3.5. It uses precomputed weight variants, a quality predictor, and NSGA-II, updating the archive with evaluated candidates. | Reducing the expensive work between profiling and a frontier is already a direct quantization research target. AMQ is weight-only; that difference must be acknowledged without treating activation support as automatic novelty. |
| Allocate evaluation resources according to the Pareto front | [Multi-objective Asynchronous Successive Halving, 2021 preprint](https://arxiv.org/abs/2106.12639), Section 4. | Pareto-aware staged evaluation is established. A small-screen/large-screen/final-evaluation schedule is not a standalone contribution. |
| Choose partial computation according to its value and cost | [Bayesian Optimization of Function Networks with Partial Evaluations, ICML 2024](https://proceedings.mlr.press/v235/buathong24a.html). The paper introduces a cost-aware acquisition for choosing a function-network node and its inputs. | Recasting the network as a computation graph and selecting partial evaluations also has direct precedent. |
| Cheap task-sensitive local candidate profiling | [NVIDIA AutoQuantize technical announcement, August 2026](https://nvidia.github.io/Model-Optimizer/announcements/20260824-autoquantize.html). It combines captured-input candidate replays with gradient scoring and constrained assignment. | Moving the novelty claim back to cheaper local sensitivity profiling would not resolve the overlap. This source is an implementation description, not a peer-reviewed paper. |

The method-level comparisons use the papers' algorithm and formulation sections where available. Numerical performance claims from those papers are not transferred to this proposal.

**Why the proposed combination is not yet sufficient**

My structural comparison is that operators map to pipeline stages, precision choices to stage parameters, and calibration state to execution dependencies. Under sufficient memory, shared work is counted once across candidate paths. Replacing generic stage parameters with bit widths does not by itself create a new scheduling result.

The remaining joint-batch, multiobjective, W/A-specific formulation is not identical to every cited algorithm. However, the conversation has not supplied a new optimizer, analysis, or quantization-specific execution mechanism that resolves this remaining problem better than a composition of those techniques. Failure to find an identical title or implementation cannot establish that difference.

**Calibration does not automatically supply the missing contribution**

Activation quantization makes reuse conditional on upstream precision, scale state, data, and observer behavior. An upstream change can invalidate downstream observations and scales. For deterministic local calibration, preserving an identical prefix and recomputing its affected suffix follows ordinary dependency tracking. For global or iterative reconstruction, reuse may be much more restricted.

A new way to transfer calibration across genuinely different upstream states, with controlled error or compelling evidence that it preserves policy ordering, could be a separate research problem. No such method has been derived or verified here. Calling ordinary cache invalidation “calibration-aware” would not establish it.

**What the workspace already contains**

- [amp/runner.py](../../amp/runner.py) implements shared-prefix evaluation in `traverse` for four stage choices.
- `accuracy_race` in the same file implements an empirical elimination rule: an unfinished candidate is rejected only once its observed mistakes exceed a fully evaluated incumbent's total mistakes on the fixed search set.

Extending these components to finer granularity can be useful engineering, but the existing behavior cannot serve as a new contribution. The runner's fixed calibration also does not demonstrate correct policy-dependent incremental calibration.

**Required novelty gate before implementation as a research project**

1. State one precise new mechanism or result that cannot be obtained just by combining the closest methods above. A theoretical result is one possible route; it is not mandatory for every publishable systems paper.
2. Specify its assumptions, the problem it solves, and the exact difference from the strongest prior algorithm. Naming a new application or combining existing techniques is insufficient evidence by itself.
3. Define a comparison that isolates that difference. The critical baseline should already have precision exchanges, staged evaluation, and the same cache implementation. Comparing only against independent full evaluations would credit known caching for the claimed innovation.
4. Evaluate the complete cost of producing a frontier: profiling, calibration, search logic, transfers, policy construction, screening, and final evaluation. Preserve equivalent resource constraints and held-out evaluation across methods.

The first two items are the current blockers to a defensible novelty claim. Experiments would subsequently establish whether the contribution is useful; speedup alone would not retroactively make the present broad algorithmic claim new.

**Final assessment**

Engineering value: plausible, unmeasured. Novel methodological contribution: not established. Publication readiness: not established. Recommendation: reject the current broad novelty claim before investing in its implementation as the research contribution.

The audit supports this negative decision for the present proposal. It does not certify that the wider research area has no publishable opportunities.
