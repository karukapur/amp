# A proposed fast mixed-precision policy search

> **Experimental update, 11 November 2026:** The signed-interaction proposal below did not consistently beat its simpler additive ablation in the real ResNet-50 pilot. See [the measured report](../../reports/legacy-pilot/REPORT.md) for results, a faster restricted exact-search alternative, certificates, and limitations. This document preserves the initial research hypothesis; it is not the final recommendation.

**Setting:** pretrained ResNet-50 image classification; signed integer weights in {4, 8} bits and integer activations in {8, 16} bits; post-training quantization (PTQ), with optional recovery after policy selection. Optimize accuracy subject to both a packed-weight-size budget and a compute budget.

**Proposal:** candidate-conditioned residual transport sketches with selective block refinement. Compute finite, joint weight/activation errors locally. Project them into a shared task-sensitive output space using a few backward probes. Search using the resulting signed vectors, audit a small number of full policies, and replace unreliable local approximations with measured residual-block responses.

This is a research proposal, not a measured ResNet-50 result. The accompanying executable checks the mechanism on a small synthetic network. Neither ImageNet accuracy nor deployment latency has been measured. The proposed research contribution is the particular combination and refinement procedure; the mathematical ingredients have precedents.

## 1. What the method needs to improve on

Relevant primary sources:

| Work | Existing ingredient | Consequence for this proposal |
|---|---|---|
| [HAWQ-V3, ICML 2021](https://proceedings.mlr.press/v139/yao21a.html) | Hardware-aware mixed-precision allocation using an integer program; integer-only graph execution. | Sensitivity-based allocation and constrained optimization are established. |
| [BRECQ, ICLR 2021](https://arxiv.org/abs/2102.05426) | Block reconstruction and second-order treatment of dependencies inside blocks. | Using residual blocks or Fisher-weighted reconstruction alone is insufficient novelty. |
| [CLADO, revised 2025](https://arxiv.org/html/2307.05657v2) | Cross-layer sensitivities, quadratic policy optimization, and an O(L²) network-evaluation procedure for profiling. | A useful comparison is retaining interactions while reducing profiling cost. |
| [InfoQ, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39039) | Global information-flow sensitivity with a training-free allocation phase. | Global sensitivity and a training-free search phase already exist. |
| [Q-WAM, September 2026 preprint](https://arxiv.org/html/2609.33269) | Shared random output probes to estimate layer observability Gramians; its derivation discards cross-layer terms under decorrelation assumptions. | Random backward probes alone are established. Retain signed candidate contractions to preserve cross terms instead of reducing them to per-layer Gramians. |

The tentative contribution is therefore **a joint W/A, signed, cross-layer response representation that can be profiled with local operator evaluations and shared backward probes, combined with audit-triggered replacement by finite block responses**. A literature search cannot establish publication-level novelty; the closest methods above need direct experimental comparisons.

## 2. Define the budgets before searching

For weighted operator i, let P_i be its weight count, MAC_i its multiply-accumulate count, w_i its weight precision, and a_i the precision of its **input** activation. Then use

    weight_bytes = sum_i P_i * w_i / 8 + metadata_and_bias_bytes
    BitOps       = sum_i MAC_i * w_i * a_i + graph_overhead

Constrain both. BitOps is a compute proxy, not a measured latency prediction. Add/requantization work should be counted separately when material. Packed storage must include scales, zero points, alignment and biases; do not report the size of a float fake-quant checkpoint as integer storage.

The four options have these idealized MAC costs:

| Weights | Input activations | BitOps/MAC | Relative to W4A8 |
|---|---|---:|---:|
| INT4 | INT8 | 32 | 1 |
| INT8 | INT8 | 64 | 2 |
| INT4 | INT16 | 64 | 2 |
| INT8 | INT16 | 128 | 4 |

W8A8 and W4A16 consume the same nominal compute per MAC but repair different error sources. Profile both. If the only budget is static weight storage, A16 has no cost in that objective: an activation tradeoff then requires a compute or activation-memory constraint.

For the [standard torchvision topology](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html), there are 16 bottleneck blocks in stages [3, 4, 6, 3], 53 convolutions including projection shortcuts, and one classifier. Treating those 54 weighted operators as independent four-way choices gives 4^54 ≈ 3.25e32 raw assignments. Graph constraints reduce that count. A parameter count near 25.6 million implies about 12.8 MB at four bits or 25.6 MB at eight bits before overhead. A parameter-weighted average of 4.8 bits is about 15.36 MB; these are arithmetic examples, not proposed measured operating points.

Specify an executable quantized graph first. Fold batch normalization, define signed/unsigned activation grids as supported by the backend, use per-output-channel weight scales where supported, and fix activation scale granularity. INT16 means integer fixed-point here, not FP16. Accumulator precision is a separate backend constraint: W8A16 can need wider accumulation than INT32 at large reductions. Include the real accumulation, saturation and requantization semantics in candidate forwards.

Assign activation bits to tensors or explicit conversion edges. Consumers sharing one stored tensor must share its format unless explicit conversions are allowed. Include projection shortcuts, residual adds, and their scale alignment. Logical input quantizers can be repeated at consumers only when this is numerically equivalent to quantizing the shared tensor once. Avoid counting a quantizer twice. These details matter more than a sophisticated search over an inaccurate graph.

## 3. Finite local errors, shared global coordinates

Choose a current anchor policy p0. A first implementation can start at W8A16 to obtain a relatively accurate reference; that anchor need not satisfy the deployment budgets. After the first feasible policy is audited, it can become the anchor for a local repair pass.

Cache floating-point teacher probabilities t_n and anchor logits z_n on N calibration images. Write p_n = softmax(z_n). At each operator or fused graph unit i, cache its actual anchor inputs h_i and output y_i.

For each of its four legal W/A choices s, evaluate only that local unit on the cached inputs:

    e_i,s,n = unit_i,s(h_i,n) - unit_i,anchor(h_i,n)

For a convolution with a fused input quantizer, this measures the finite response of Conv(Q_w(W), Q_a(h)), including the product between weight and input-activation quantization errors. Do not infer the W4A8 response by adding independent W4 and A8 scalar penalties. Fixed clipping, rounding, fused nonlinearities, and requantization semantics belong in the local evaluation.

Let J_i,n map a perturbation of unit i's output to the final logits along the anchor graph. The working approximation is

    delta_z_n ≈ sum_i J_i,n * e_i,s_i,n.

The local errors are finite measured quantities. Their propagation and superposition are approximate. Changes in downstream inputs, ReLU gates, clipping and rounding bins can invalidate this model. This distinction motivates the audit and block refinement below.

### Task-sensitive probes

Use teacher KL as a continuous search objective:

    L = mean_n KL(t_n || softmax(z_n)).

Its logit gradient is p_n - t_n and its logit Hessian is

    H_n = diag(p_n) - p_n p_n^T.

For each example and probe r, draw independent standard Gaussian epsilon_n,r and construct

    u_n,r = sqrt(p_n) * epsilon_n,r
    q_n,r = u_n,r - p_n * sum_c u_n,r,c.

Then E[q q^T] = H_n. All coefficients are detached during backpropagation. Backpropagate the scalar sum_n q_n,r^T z_n. One reverse sweep gives g_i,n,r = J_i,n^T q_n,r at **every** unit's output.

Contract these gradients with every locally measured error:

    v_i,s[n,r] = dot(g_i,n,r, e_i,s,n) / sqrt(N * R).

Flatten the N×R entries to make one signed vector per choice. The example and probe indices must align across every unit. Independent sketches or per-layer permutations would destroy the intended cross-layer products. Do not sum the signed responses over examples before squaring: different images must not cancel one another.

Also perform one backward pass of the actual mean KL loss and store

    a_i,s = sum_n dot(gradient_of_mean_L_wrt_y_i,n, e_i,s,n).

The gradient already includes 1/N. The first-order term is necessary after reanchoring: quantized anchor predictions generally differ from the teacher. Dropping it can penalize upgrades that improve the anchor. With labeled calibration data, replace the teacher target by a convex mixture of teacher probabilities and one-hot labels; the same H_n and probes still apply.

### The policy surrogate

Use

    L_hat(policy) = L(anchor)
                  + sum_i a_i,s_i
                  + 0.5 * ||sum_i v_i,s_i||².

Expanding the square gives both individual penalties and signed interactions:

    0.5 * sum_i ||v_i,s_i||² + sum_{i<j} dot(v_i,s_i, v_j,s_j).

A positive dot product means reinforcement in task-sensitive output directions; a negative product means partial cancellation. For example, signatures (1,0) and (-1,0) each have unit squared norm, but their combined error is zero in this approximation. A scalar sum cannot represent that.

For a fixed policy and fixed linearized transport, the randomized quadratic term is unbiased for

    (1 / (2N)) * sum_n delta_z_n^T H_n delta_z_n.

For independent Gaussian probes, let d_n = delta_z_n^T H_n delta_z_n. The estimator variance is sum_n d_n² / (2*N²*R). Its relative standard error is sqrt(2/(R*N_eff)), where N_eff = (sum_n d_n)² / sum_n d_n². Increasing the image count helps less if a few images dominate the error. This fixed-policy statement does not protect against an optimizer selecting a policy that exploits sketch noise; independent audits remain necessary. Start with a small probe count and increase it only if finalist rankings are unstable.

It is **not** an unbiased estimate of actual finite quantized loss or classification accuracy. This is a Gauss-Newton output approximation, not the complete network Hessian. If p_n != t_n, it also omits terms involving second derivatives of the logits. Quantized-anchor Jacobians require a smooth transport convention; a practical choice is frozen quantized weights and a clipped straight-through derivative for activation rounding. That surrogate derivative is a limitation of the model, not a derivative of true integer rounding.

## 4. Search without running the network

Use a one-hot variable x_i,s for each candidate. Let V have signature vectors as columns. The objective is

    minimize a^T x + 0.5 * ||V x||²
    subject to one option per unit, size/compute budgets, and graph constraints.

The continuous quadratic matrix V^T V is positive semidefinite. The integer problem is still combinatorial. A time-capped mixed-integer quadratic solver is one implementation; it does not guarantee a globally optimal answer within the cap. A simpler alternative is feasible greedy initialization followed by budget-preserving one-site moves and two-site exchanges from several feasible starts. Always include an independently optimized additive-sensitivity policy in the candidate pool so that a poor interaction model can be rejected during auditing.

For a cached current sum z = sum_i v_i,s_i, a change with d = v_i,new - v_i,old has score change

    delta = a_i,new - a_i,old + z^T d + 0.5 * ||d||².

The same formula handles a two-site exchange by adding the two signature changes. Moves and exchanges must check all budgets and tied-format constraints; a simple gain/cost ranking is insufficient for multiple budgets. This is O(NR) arithmetic per scored move and requires no model inference. Resource counters can be updated with the affected graph units.

With N=256, R=8 and 54×4 candidate columns, float32 signatures occupy approximately 1.8 MB. This is only signature storage: calibration activations, local errors, the teacher, and autograd state can be much larger. Profile in microbatches. Quantized weights can be prepared once; retain local errors for a microbatch across the R reverse sweeps, contract them immediately, and discard the batch. Recomputing every candidate on every reverse sweep would undermine the proposed speed advantage.

## 5. Audit and selectively repair the approximation

Use separate images for observer/range calibration, response profiling, policy selection, and final reporting as data availability permits. The final ImageNet validation set must not be repeatedly optimized against.

1. Generate a small diverse shortlist, initially about 8–12 feasible policies. Include the additive policy, nearby exchanges, and multiple solver incumbents. A shortlist of numerical near-duplicates provides little protection. If most candidates differ only in tiny A8/A16 effects, reserve slots for different weight patterns, while retaining activation variants where they are material.
2. Run these policies through the complete fake-quant graph on a disjoint selection set. Compare teacher KL, labeled loss if labels are available, and prediction flips. Log clipping and ReLU changes by block. Do not infer a sub-percentage-point accuracy improvement from 128 calibration images.
3. Keep the best **measured** feasible policy. A lower surrogate score alone never replaces the incumbent.
4. If prediction/ranking disagrees materially with measurement, refresh the transport model around that policy. Preserve the teacher and recompute both the linear and quadratic terms.
5. Where nonlinear interactions remain important, merge selected graph units into a full residual bottleneck factor. For block B, measure

       e_B,s = B_s(h_B,anchor) - B_anchor(h_B,anchor)

   and project at its output using the same probes. Replace the original internal factors with this block factor; adding it on top would count their errors twice. It includes the shortcut, projection, add and quantization boundaries. Condition on legal input/output formats as required.
6. Begin with the incumbent block pattern, uniform low/high patterns, single-site neighbors and promising paired changes. Expand only the blocks that need it. An identity bottleneck with three weighted operators has 64 four-way patterns; a projection bottleneck with four has 256, before format constraints. Enumerating every block at the outset can erase the time savings.

Block diagnostics can use observed clipping/gate changes as heuristics. A stronger diagnostic replays a candidate block and the anchor block on the candidate's observed input, then compares their finite response with the response at the anchor input. That extra local work must be included in the search-time total. It indicates context dependence, not a proof of a block's global causal responsibility.

Use an independent small set of probes to audit shortlist scores if sketch noise is suspected. Real policy evaluation catches combined sketch and model errors; independent probes help distinguish them. If interaction modeling keeps failing, return the measured additive incumbent and report that outcome. Do not keep expanding the search indefinitely.

To reduce selection cost, a fixed racing schedule can evaluate twelve policies on 64 images, keep six and extend to 256 images, then keep three and extend to 1,024. Reusing the nested batches requires 4,224 image-policy forwards, compared with 12,288 for evaluating all twelve on 1,024 images. Early elimination can discard the eventual best policy, so preserve a baseline candidate and assess this tradeoff experimentally. This schedule is a computational allocation heuristic, not a statistical guarantee.

Optional robustness: shrink cross terms using

    0.5 * [(1-rho) * ||sum_i v_i||² + rho * sum_i ||v_i||²],  0<=rho<=1.

This preserves the diagonal penalties and reduces cancellation dependence. It stays positive semidefinite. Treat rho as an ablation or selection-set choice, not an asserted universal default.

## 6. Why this can reduce search time

With four local options, three alternatives to the anchor require roughly three total network-equivalents of **local operator computation**, when summing over the graph. They do not require three full network runs per layer. That is an arithmetic-work estimate: quantization kernels, tensor movement and launch overhead can change wall time substantially.

Let F mean one forward over the profiling set and B one activation-backward sweep over that same set. A basic pass costs approximately

    teacher/anchor work + 3F_local + (R+1)B + shortlist evaluation + solver time.

For an illustrative accounting with teacher/anchor=2F, R=8, B=2F, and twelve shortlist evaluations on an equally sized set, the sum is about 35F. Additional observer calibration, larger selection sets, reanchoring, block refinement and final evaluation are extra. This is not a measured speedup.

A naive one-layer-at-a-time sensitivity sweep with 54 operators and three non-anchor modes takes 162 full-network evaluations before policy validation. Dense pairwise perturbation profiling scales worse. Conversely, inexpensive additive methods may already be faster than the proposed method; the comparison must be accuracy at matched search time, not a claim of superiority to every baseline. The proposed advantage is cheap **interaction-aware** profiling without a trained supernet or O(L²) full-network pair tests. Searching another budget can reuse the same initial signatures, although faraway budgets may require new anchors and independent validation.

## 7. Controlled mechanism experiment

Run:

    python3 transport_sketch_demo.py

The script uses installed PyTorch, NumPy and SciPy. It makes no downloads and trains no model. It exhaustively evaluates 1,024 policies on an untrained five-block residual MLP with the requested integer widths, for three random seeds and two anchors (W8A16 and W4A8). There are 256 feasible policies per case under toy equal-sized weight and activation-storage budgets. Those toy budgets are not a ResNet BitOps model. The classifier remains floating point.

The script compares the coupled and additive surrogates using identical errors and probes. Both keep the linear term; the additive version removes cross terms. It also calculates explicit tiny-network Jacobians as an independent transport check. This expensive check is deliberately limited to the synthetic validation and is not part of the proposed production search.

Observations are recorded in `transport_sketch_results.json`. Policy-rank correlation on the profiling images averaged about **0.964** for the coupled score and **0.915** for the additive score. Sketch scores versus explicit transport scores averaged about **0.982**. The coupled selection had lower actual profiling-set KL in four of six cases, tied in one and lost in one. On 256 disjoint images, however, these directly selected policies won in only two cases, tied in one and lost in three. These six cases reuse three random networks across two anchors; they are not six independent benchmark trials.

The script also compares three twelve-policy shortlists: a mixture of coupled/additive rankings, an additive-only ranking, and random feasible policies. Each uses the same 256 selection images and reports its selected policy on a further 1,024 images. A small diversity rule permits at most two candidates with the same weight pattern. In this experiment, the mixed and additive-only shortlists selected the same policies in all six cases. **The test does not establish an end-to-end quality advantage over a well-validated additive method.** It demonstrates better proxy ranking, exposes the generalization problem, and shows why the additive control and full-policy selection are essential. The method remains worth testing on ResNet-50, especially where measured interactions matter; it should be rejected there if it does not earn its profiling cost. Twelve evaluations per method equalize shortlist evaluation counts, not complete wall time.

Checks cover the Fisher probe covariance, the softmax constant-shift nullspace, signed cancellation, zero anchor errors, the exact anchor score and the cached swap-score identity. This establishes that the implementation behaves consistently with the derivation in a small setting. It establishes neither trained-model accuracy nor a ResNet-scale speed advantage.

Using prefixes of the same eight probes, average policy-rank correlations were 0.900 with one probe, 0.911 with two, 0.956 with four and 0.964 with eight. This suggests a four-probe starting point for the next experiment, with escalation based on independent ranking checks. It does not establish that four probes suffice for ResNet-50.

## 7a. Minimal search loop

```text
teacher_targets = run_float_model(profiling_images)
anchor = W8A16
incumbent = none
repeat within a fixed time budget:
    cache anchor inputs, outputs and logits in microbatches
    measure each unit's four joint W/A errors on its anchor inputs
    compute one loss-gradient sweep and four shared Fisher-probe sweeps
    contract errors with gradients to obtain a and V
    propose feasible coupled and additive policies under size/compute limits
    race a diverse shortlist on disjoint selection images
    keep the best measured feasible policy as incumbent
    stop if remaining time or measured improvements do not justify refinement
    anchor = incumbent
    replace selected unreliable unit groups with finite block-response factors
freeze policy; optionally recover its accuracy; report on final validation data
```

The first pass needs no block replacement. A practical implementation should cap both solver time and audit/refinement time, and return its measured incumbent when the cap is reached. A fixed time budget makes the quality-versus-search-time comparison reviewable.

## 8. The ResNet-50 experiment that would decide whether it is useful

Suggested starting configuration, to be tuned after measurement:

- One exact pretrained checkpoint and preprocessing recipe; report its own floating-point top-1.
- Observer calibration on 512–1,024 representative training images. Profile on 128–256 images, with microbatches sized to memory and 4–8 probes per image. Use a disjoint selection subset, initially 1,024–2,048 images. Avoid mining only hard examples; if importance sampling is used, retain proper weights.
- Fixed deployable quantizers for all methods: per-channel W4/W8 and static activation A8/A16, with the same clipping optimization, BN folding and bias treatment. Verify integer graph semantics separately from fake quant.
- Start with two or three resource budgets, then trace a Pareto frontier. Use both a packed-size constraint and a normalized BitOps constraint; choose values inside the feasible range after profiling the actual graph. Do not prescribe a layer policy in advance.
- Compare uniform W4A8 and W8A8 where feasible, random feasible search at matched wall time, a joint-W/A additive allocator with the same calibration, HAWQ-style allocation, and CLADO-style interaction modeling. Include BRECQ reconstruction as a matched optional recovery stage. Add a recent fast global-sensitivity baseline such as InfoQ if reproducible code/settings are available.
- Ablate cross terms, joint-versus-separate W/A errors, probe count, reanchoring, and selective block refinement. For the last ablation, compare at equal total time: extra block profiling is not free.
- Plot best measured held-out quality versus elapsed search time; report final ImageNet top-1, packed bytes, BitOps, peak calibration memory and the complete calibration/search/recovery time. Run multiple calibration seeds. Report the additive fallback and failures.

The strongest result would be a frontier showing a consistent accuracy benefit at the same budgets and lower total search time than a competitive interaction-aware baseline. A practical goal could be to match a much longer search within about 0.2 percentage points while reducing total search time severalfold. Those are experimental success criteria, not forecasts.

Only after selecting and freezing a policy should optional recovery run: bias correction, chosen-block reconstruction, or a short quantization-aware fine-tune. Use the same recovery budget for baselines. If recovery substantially changes weights or ranges, the old response sketches are stale; report that cost and rebuild only if further policy search is needed.

## 9. What could falsify the proposal

- Most errors may be effectively independent for this checkpoint and these widths. Then the additive allocator can achieve the same accuracy with less work.
- Cross-layer cancellation on calibration data may not generalize. Independent selection, probe checks, shrinkage and multiple seeds expose this.
- W4 can move the network outside the local transport regime. Reanchoring and selective block factors can help, but if many blocks require enumeration the proposed time advantage can disappear.
- KL may rank policies well yet fail to preserve labeled top-1. Labeled selection and final evaluation must decide.
- A16 may provide little benefit when activation calibration is good; allocating no A16 tensors is a valid policy. A16 cannot repair a weight-dominated error.
- The compute proxy may disagree with actual latency. Model-size/BitOps claims should stay distinct from hardware speed claims.

The recommended first implementation is the local joint-error table, 4–8 shared Fisher probes, the signed-vector objective, an additive incumbent, and a small measured shortlist. Add block refinement only when the audit demonstrates a need. This keeps the initial experiment small enough to determine whether the core interaction representation earns its cost.
