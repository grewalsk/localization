# Convergent Validity of Localization Signals for LLM Inference

**Do the localization methods we actually deploy agree on where information lives, and does their disagreement predict inference-time failure?**

---

## 1. Summary

A growing family of inference methods is justified by an interpretability claim of the form "the information relevant to this task lives at component or token X." KV-cache eviction keeps heavy-hitter tokens; head-aware compression keeps retrieval heads; SAE and probe based detectors read truth or error from specific positions. Each method is validated against its own downstream task. Almost none are validated against each other. This project asks whether these localization signals agree *per instance*, validates each against a per-instance causal oracle, and tests whether cross-method disagreement is a training-free predictor that a real compression method will break that instance. The central deliverable is not a method ranking; it is the disagreement-predicts-fragility result, with the agreement structure as the diagnostic that explains it.

## 2. Motivation and problem statement

The field has two uncomfortable precedents. Hase et al. (2023) showed that causal-tracing localization does not predict the best layer to edit in ROME; the editing works for reasons orthogonal to the localization claim. Kantamneni et al. (2025) showed that SAE probes do not beat logistic regression on raw activations across four probing regimes; the "interpretability" content adds nothing over the raw signal on that task. Both results bound a single method's localization claim. Neither asks the meta-question: if you line up the localization methods that practitioners deploy for inference and ask them to point at the tokens or components that mattered for a specific input, do they even agree with each other?

If they agree and that agreement tracks a causal oracle, the field's localization claims are mutually consistent and the deployed methods are interchangeable. If they disagree, "the model stores information at X" is a method-relative artifact rather than a fact about the model, which is the strongest generalized form of the Hase and Kantamneni worry. The point of attack is that nobody has run this on the objects that drive real inference; existing cross-method work targets stylized circuit-discovery tasks (see Section 7).

A second, orthogonal conflation motivates the substrate design in Section 5.2. The literature mixes two distinct localization questions: *input attribution* ("which context token caused the answer") and *internal-confidence localization* ("where, or whether, the model internally represents that it knows the answer"). KV heavy-hitters and retrieval heads answer the first; ITI truth-directions and Orgad's exact-answer-token probes answer the second. Treating these as one "where does the information live" question is a category error. Making the distinction explicit is part of the contribution.

## 3. Research questions

1. **Within-substrate agreement.** For commensurable localization signals (defined in 5.2), how strongly do per-instance importance rankings agree across methods, and how does agreement vary with task difficulty and context length?
2. **Causal grounding.** Does each method's importance track a per-instance causal oracle (activation or attribution patching)? This is the test that separates "the tools are noisy" from "localization is genuinely plural."
3. **Disagreement as a failure predictor.** Does pre-measured cross-method disagreement predict that a fixed-budget KV-compression method flips a correctly-answered instance to incorrect, beyond what trivial difficulty proxies (length, needle depth, model confidence) already predict?
4. **Attribution versus confidence.** Do input-attribution methods and internal-confidence methods localize to systematically different places, and does the gap between them carry signal (for example, about hallucination)?

## 4. Related work and positioning

**Deployed localization signals (the objects under study).** Heavy-hitter KV eviction: H2O (Zhang et al., 2023, arXiv:2306.14048), Scissorhands (Liu et al., 2023, arXiv:2305.17118), SnapKV (Li et al., 2024). Retrieval-head attention: Wu et al. (2025, arXiv:2404.15574) and its compression descendant DuoAttention (Xiao et al., 2025, arXiv:2410.10819). Internal-confidence localization: ITI (Li et al., 2024, arXiv:2306.03341) and Orgad et al. (2025, arXiv:2410.02707). SAE and transcoder features as a localization substrate: Gemma Scope (Lieberum et al., 2024, arXiv:2408.05147), Gemma Scope 2 (DeepMind, 2025), Llama Scope (He et al., 2024, arXiv:2410.20526), Qwen-Scope (2026).

**Cross-method comparison and its limits.** MIB (Mueller et al., 2025, arXiv:2504.13151) and the BlackboxNLP 2025 shared task (arXiv:2511.18409) compare localization and featurization methods on a fixed benchmark and find, notably, that SAE features are not better than raw neurons for causal-variable localization. This is the closest existing work and the most important to differentiate from (Section 7). MIB scores each method against one privileged causal target on stylized, single-token circuit tasks; it does not work per instance on real long-context QA, does not include the KV or retrieval-head inference machinery, and structurally cannot produce the "multiple methods each causally real but mutually disagreeing" finding because every method is graded against the same target.

**Decodability is not causality.** A linear direction being decodable does not mean the model uses it. The counting-ViT study (arXiv:2510.09794) makes this explicit, and the activation-patching best-practices work (Zhang and Nanda, 2024, arXiv:2309.16042) shows the causal oracle itself is hyperparameter sensitive, which constrains our protocol (Section 5.5).

**The classic precedent.** The disagreement problem in explainable ML (Krishna et al., 2022) and feature-attribution disagreement studies (for example Kamp et al., arXiv:2310.05619) established that saliency methods disagree on the same input. That literature lives on gradient and SHAP attributions over classifiers. It has never been run on the modern objects that drive LLM inference, and it never coupled disagreement to a deployment failure mode.

**The gap, stated in one breath.** Per-instance, not aggregate. Deployed inference objects, not stylized circuits. Disagreement-forecasts-failure, not method-ranking.

## 5. Approach

### 5.1 Objects under study

We compare the localization signals that are actually wired into inference methods, not arbitrary interpretability probes:

1. Accumulated-attention token importance (H2O / Scissorhands operationalization).
2. Retrieval-head attention mass per token (Wu retrieval heads, identified once via a needle-retrieval score).
3. Transcoder feature attribution back to context tokens.
4. Internal-confidence localization: ITI truth-direction head selection and Orgad exact-answer-token probing.

### 5.2 Two substrates plus the attribution/confidence split

We never report a single agreement number across incommensurable objects. Agreement is defined only within commensurable substrates, with the projection stated explicitly.

- **Token-attribution substrate** ("how much did context token *t* cause the answer"). Members: accumulated-attention (native), retrieval-head attention summed onto each token, transcoder feature attribution to tokens. Heavy-hitter and retrieval-head signals both read attention, so their mutual agreement is partly mechanical; the informative comparisons are across families (attention versus transcoder).
- **Component substrate** ("which heads do the work"). Members: retrieval heads (native), ITI truth-carrying heads (native), transcoder attention-to-feature loadings. Heavy-hitters are excluded (token-level only).
- **Internal-confidence, answer-position substrate.** ITI and Orgad signals read the answer position and say nothing about which context token mattered. They get their own analysis (RQ4) and are explicitly kept out of the token-attribution matrix. ITI still contributes its selected heads to the component substrate.

### 5.3 Operationalizing each method into a comparable score

Each token-substrate method produces a per-instance importance vector over context tokens.

- **Accumulated attention.** Sum (or last-window cumulative, per Scissorhands) attention received by each context token, aggregated across heads and layers.
- **Retrieval-head attention.** Identify retrieval heads once via a synthetic needle-retrieval score, then per instance sum those heads' attention onto each context token.
- **Transcoder attribution.** Identify the feature(s) most associated with the answer (active at the answer position, or top by answer-logit attribution), then attribute back to context tokens via gradient-times-activation, verified against feature ablation on a subsample.

Component-substrate methods produce a per-instance (or global) score over heads via the same identification procedures plus difference-in-means head attribution for ITI.

### 5.4 Agreement metrics

- Spearman rank correlation between method importance vectors, averaged over instances, reported as a pairwise heatmap per substrate.
- Top-*k* Jaccard overlap for *k* in {gold-span size, 5, 10, 20} and rank-biased overlap for a top-weighted view.
- Agreement with the gold answer span where available: precision@*k* and AUROC of importance against gold-span membership.
- Per-instance disagreement score D(x) = 1 minus the mean pairwise Spearman across methods (within substrate). D(x) is the quantity carried into the fragility test.

### 5.5 Causal oracle and controls

For RQ2 and to separate noise from plurality, we compute a per-instance causal effect for each token and head.

- **Effect.** E(t) is the drop in the answer log-probability when token *t*'s key/value is **masked** from attention at every layer (Definition 2; the primary oracle, **[Update 2026-06-05]** superseding the earlier "ablated or patched" framing — masking is read as an upper-bound proxy for eviction's per-token loss, not the identical operation). The analogue for heads is the change under head ablation. Symmetric token-swap is a robustness check on the ranking, not the primary oracle.
- **Scaling.** Per-token masking across long contexts is linear in context length and infeasible at scale, so we use attribution patching (Syed et al., 2023) for the full sweep and verify it against true masking on a 5 to 10 percent subsample, reporting the **Spearman** rank correlation between the linear estimate and the true effect (ranking, not magnitude, is what every downstream use needs; gate 11.2a requires Spearman ≥ 0.8, else the pinned 150-instance exact-masking fallback).
- **Method-versus-oracle fidelity.** Per instance, correlate each method's importance with E. High individual fidelity plus low mutual agreement is the strong plural-localization finding; the activation-patching best-practices result forces us to fix and report the patching configuration (corruption type, metric, granularity) and to show robustness to it, otherwise the oracle is as arbitrary as the methods it judges.

### 5.6 The deployment payoff

The result with teeth. Take instances answered correctly under the full cache. Apply H2O and SnapKV at a fixed budget (sweep the budget). Label an instance "flipped" if it goes correct to incorrect. Predict flips from D(x) via logistic regression and report AUROC. The non-negotiable control: refit with confound covariates (context length, needle or gold-span depth, answer-token entropy as a confidence proxy) and report the incremental AUROC and likelihood-ratio test of D(x) over the confound-only model. The claim is only interesting if disagreement adds signal beyond trivial difficulty proxies; otherwise we have rediscovered that hard questions break under compression. If it holds, D(x) is an a-priori, training-free flag for compression fragility, and the obvious application is routing aggressive compression away from high-disagreement instances.

### 5.7 Models, datasets, tooling

- **Primary model.** Llama-3.1-8B-Instruct, chosen because the retrieval-head and KV-compression lineages overwhelmingly use Llama and Llama Scope provides SAEs and transcoders on every sublayer for all 32 layers.
- **Replication model.** Gemma-2-9B-it with the original Gemma Scope (SAE-only), to show the disagreement structure is not model-specific. **[Errata 2026-06-02]** Gemma 2 has no transcoder suite (transcoders only exist for Gemma 3 / Gemma Scope 2); the transcoder-based signals are dropped in replication and reported as a finding (§11.2c / §13.6). Qwen-Scope gives a third option for the cross-model extension.
- **Why transcoders, not residual SAEs.** Transcoders are built for clean circuit attribution (freeze them and the input-to-feature path stays interpretable) and they sidestep part of the irreducible-reconstruction-error objection.
- **Datasets.** Synthetic needle-in-a-haystack for controllable gold spans and needle-depth confounds; HotpotQA for multi-hop with supporting-fact annotations as gold spans; LongBench subsets for realistic long-context distribution. A single-hop set (Natural Questions or TriviaQA) gives the easy-versus-hard contrast. **[Errata 2026-06-02]** Canonical HF identifiers (verified live; the originals were renamed): `hotpotqa/hotpot_qa` (`distractor`), `zai-org/LongBench` (QA subsets, `trust_remote_code=True`), `mandarjoshi/trivia_qa` (`rc`). See engineering-addendum §10 for the full schema errata.
- **Mandatory SAE control.** Report the answer-relevant variance captured by the chosen transcoder features, so a low transcoder-versus-others agreement can be attributed to genuine plural localization rather than a lossy feature dictionary. This directly answers the Heap et al. (2025) "random-model SAEs look interpretable" and the SAE-irreducible-error concerns.

### 5.8 Cost and mitigations

Run the oracle-validated core at 1k to 4k context (the point does not require 40k). Patch at sentence or span granularity rather than per token. Use attribution patching for the sweep with subsample verification. These are the standard moves for this exact scaling problem and keep the project on a single-researcher, single-GPU footing for the core results.

## 6. Outcome space and interpretation

1. **Methods disagree and all track the oracle poorly.** The tools are noisy. Weakest outcome, still informative about the reliability of deployed localization signals.
2. **Each method individually tracks the oracle but they disagree with each other.** Multiple distinct, causally real localizations exist and each method captures a different one. This is the strong, non-obvious finding and the one MIB's framing cannot produce.
3. **One method tracks the oracle, others do not.** A ranking of the deployed methods on real long-context QA, a Kantamneni-style verdict on objects Kantamneni never tested.

In all three, the Section 5.6 fragility result is an independent payoff that does not depend on which case obtains.

## 7. How this differs from MIB (state it every time)

MIB is the obvious reviewer challenge and the answer must be reflexive. MIB compares methods against one privileged causal target on stylized, fixed-counterfactual circuit tasks and asks which method is most faithful. This project works per instance on real long-context QA, includes the KV and retrieval-head inference machinery that MIB omits, and asks whether the deployed signals agree with each other and whether their disagreement forecasts a real compression failure. We use MIB's track distinction (circuit localization versus causal-variable localization) as scaffolding for our substrate split and cite it as the foundation, not the competitor.

## 8. Risks and mitigations

- **"How is this not MIB."** Mitigated by the one-breath differentiation (per-instance, deployed objects, disagreement-forecasts-failure) baked into the framing and repeated in the writeup.
- **Disagreement is just noise.** Mitigated by the causal oracle; case 2 versus case 1 is decided by individual method-to-oracle fidelity, not by agreement alone.
- **Oracle is arbitrary.** Mitigated by fixing and reporting the patching configuration and showing robustness, following the activation-patching best-practices guidance.
- **Transcoder attribution is lossy.** Mitigated by the captured-variance control and by treating the transcoder as one substrate member rather than the arbiter.
- **Fragility result is a difficulty proxy in disguise.** Mitigated by the incremental-AUROC test over length, depth, and confidence covariates.
- **High agreement makes the paper boring.** Mitigated because the oracle ranking and the fragility flag are each publishable on their own, so the floor is non-empty.

## 9. Milestones and phasing

- **Phase 0.** Build the harness; reproduce each method's localization on the primary model; sanity-check on synthetic needle tasks where the gold span is known.
- **Phase 1 (can't-null core).** Within-substrate agreement matrices and gold-span precision on synthetic needle plus HotpotQA.
- **Phase 2 (causal grounding).** Attribution-patching oracle with subsample verification; place each method against it; classify the outcome (Section 6).
- **Phase 3 (the payoff).** Compression-fragility prediction with the confound-controlled incremental-AUROC test.
- **Phase 4 (replication).** Repeat the core on Gemma-2-9B-it.
- **Stretch.** Cross-model universality of the disagreement structure; deepen the component substrate.

## 10. What we want to do after

The harness and oracle built here are the substrate for several follow-on projects, in rough priority order.

1. **Component substrate to safety.** The component-agreement protocol is most of the infrastructure for asking whether safety-critical heads coincide with the heads KV compression deprioritizes. Computing safety-head importance, retrieval-versus-streaming partition, and their overlap in one model, then testing whether compressing the streaming-and-safety overlap erodes refusal, turns the diagnosis into a head-level mechanism for compression-induced safety regression and a targeted fix (protect those heads).
2. **Intervention-by-efficiency compatibility.** With the oracle and a deployment harness in hand, test whether early-exit and KV eviction silently disable activation steering. A token that exits before the steering layer never receives the intervention; the sharpest single number is how much of ITI's TruthfulQA gain survives once early exit is on. Build the steering-by-efficiency compatibility matrix and the two fixes (a steering floor on exit depth; re-applying the direction at the exit layer).
3. **Reasoning models.** Does localization plurality grow with long chain-of-thought? Per-sentence attribution over reasoning traces, connecting to the thought-anchors line, asks whether disagreement is larger where the model is reasoning rather than recalling.
4. **Cross-model universality.** Run the disagreement protocol on Llama, Gemma, and Qwen (all now have open SAE and transcoder suites). If the disagreement structure is model-invariant, the epistemics claim strengthens from "true of this model" to "true of transformers as trained."
5. **Productionize the fragility flag.** Turn D(x) into a runtime router that sets a per-instance compression budget, and measure end-to-end memory savings at fixed accuracy against uniform-budget baselines.
6. **Contribute back to MIB.** Offer the per-instance, deployment-grounded protocol as a complementary track or as evidence, and report whether deployed-object localization behaves differently from MIB's stylized-task localization.

## 11. Selected references

- Hase et al., Does Localization Inform Editing?, NeurIPS 2023, arXiv:2301.04213.
- Kantamneni et al., Are Sparse Autoencoders Useful?, ICML 2025, arXiv:2502.16681.
- Mueller et al., MIB: A Mechanistic Interpretability Benchmark, ICLR 2025, arXiv:2504.13151.
- BlackboxNLP 2025 Shared Task, arXiv:2511.18409.
- Zhang and Nanda, Towards Best Practices of Activation Patching, ICLR 2024, arXiv:2309.16042.
- Syed et al., Attribution Patching Outperforms Automated Circuit Discovery, 2023.
- Counting ViTs: Causality is not Decodability, arXiv:2510.09794.
- Krishna et al., The Disagreement Problem in Explainable Machine Learning, 2022; Kamp et al., arXiv:2310.05619.
- Wu et al., Retrieval Head Mechanistically Explains Long-Context Factuality, ICLR 2025, arXiv:2404.15574.
- Xiao et al., DuoAttention, ICLR 2025, arXiv:2410.10819.
- Zhang et al., H2O, NeurIPS 2023, arXiv:2306.14048; Liu et al., Scissorhands, NeurIPS 2023, arXiv:2305.17118; Li et al., SnapKV, 2024.
- Li et al., Inference-Time Intervention, NeurIPS 2024, arXiv:2306.03341; Orgad et al., LLMs Know More Than They Show, ICLR 2025, arXiv:2410.02707.
- Lieberum et al., Gemma Scope, 2024, arXiv:2408.05147; Gemma Scope 2, DeepMind, 2025; He et al., Llama Scope, 2024, arXiv:2410.20526; Qwen-Scope, 2026.
