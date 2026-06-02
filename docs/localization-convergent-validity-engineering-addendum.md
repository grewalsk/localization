# Engineering Addendum: Convergent Validity of Localization Signals

**Companion to `localization-convergent-validity-spec.md`. This document closes the deterministic engineering gaps in the scientific spec and marks the non-deterministic forks as explicit human-judgment checkpoints.**

---

## 0. Purpose and operating model

The scientific spec describes *what* to measure and *why* the result is publishable under any outcome. It is deliberately silent on the engineering, and that silence is dangerous for this particular project. The research design has a can't-null property (Section 6 of the spec), but that property protects the *science*, not the *implementation*. The failure mode here is not a crash. It is code that runs end to end, emits a clean heatmap and a plausible AUROC, and is wrong because an SAE was wired to the wrong hook point, attention was read from the wrong token set, the chat template shifted every index by a constant, or attribution patching silently disagreed with the true effect it was supposed to approximate. None of those throw an exception. An autonomous agent has no oracle to check itself against, so it will hand back a finished-looking artifact you cannot trust.

This addendum is therefore written for a specific workflow: **a coding agent builds components, and a human validates at phase boundaries using the acceptance tests in Section 11.** The agent can own the harness, data loading (given exact identifiers), metric computation (given exact definitions), the statistical tests, and the plots. The agent cannot be trusted to wire the interpretability instrumentation correctly without a human who already knows what the numbers should look like. The acceptance tests exist to make "what the numbers should look like" machine-checkable wherever possible, so that "it ran" is never the only signal of success.

Every section below is one of two kinds. Deterministic sections pin a choice the agent must not deviate from (versions, identifiers, formulas, algorithms). Checkpoint sections flag a decision the agent must surface to you rather than resolve on its own; these are collected again in Section 13.

A note on the prime directive: wherever this document recommends an external library or repo by name, the agent must treat the name as a starting point to *verify against the current ecosystem*, not as gospel, because library names and APIs drift. The corresponding acceptance test (reproduce a published number) is what actually guarantees correctness, and it holds even if a recommended library has been renamed or superseded.

---

## 1. Environment and dependency pinning

### 1.1 Hardware

A single 80GB GPU (A100 80GB or H100) is sufficient for the entire core of this project. The retrieval-head detection reference confirms a single 80GB card detects heads up to 50K context; everything else in this project runs at 1K to 4K context for the oracle-validated core, which is lighter. The replication model (Gemma-2-9B) also fits. Multi-GPU is not required and would add coordination complexity for no scientific gain. Budget for disk: SAE/transcoder checkpoints for one model across the layers you use are on the order of low tens of GB; activation caching is on-the-fly, not stored.

### 1.2 Version pins and the reference-repo conflict

The single most important environment fact: the canonical retrieval-head repository (`nightdessert/Retrieval_Head`) pins `transformers==4.37.2`, which predates Llama-3.1 and **cannot load the target model**. Do not pip-install that environment. Instead, port the algorithm (Section 5.2) onto a modern stack:

```
python        >= 3.10
torch         >= 2.1
transformers  >= 4.44        # Llama-3.1 support landed in 4.43; 4.44+ is safer
TransformerLens >= 2.0       # primary instrumentation layer; see Section 2
sae_lens      (current)      # loads Llama Scope SAEs onto TL hook points
nnsight       (current)      # fallback instrumentation only; see Section 2
scipy, scikit-learn, statsmodels   # agreement metrics and the logistic-regression flip test
flash-attn    (optional)     # speed only; never used on attention-reading passes
```

Pin exact resolved versions into a `uv` lockfile (or `requirements.txt` with hashes) once the environment is first known-good, so the parity acceptance test (11.0) is reproducible. The `flash-attn` caveat is load-bearing and recurs throughout: FlashAttention does not return attention weight matrices, so any pass that reads attention must use `attn_implementation="eager"`. The reference retrieval-head code handles this by caching the prefill with FlashAttention and then re-running with eager attention for the decoding steps it scores; if you instead use TransformerLens, this is moot because TL uses its own attention implementation that always exposes the pattern.

---

## 2. Instrumentation stack (checkpoint, with a default)

Three viable backbones exist and they are not interchangeable at 8B scale.

**TransformerLens (`HookedTransformer`)** reimplements the model with standardized hook points (`blocks.{l}.hook_resid_post`, `blocks.{l}.attn.hook_pattern`, `blocks.{l}.attn.hook_z`, `blocks.{l}.mlp.hook_pre`, and so on). Its activation-patching and attribution-patching utilities are first-class and heavily documented, and SAELens loads Llama Scope SAEs directly onto these hook points. The costs are higher memory and *numerical drift from Hugging Face*: TL folds layernorm and reprocesses weights, so its logits are not bit-identical to HF. For activation analysis this is usually fine, but it must be verified, not assumed (test 11.0).

**Hugging Face + nnsight** runs the real model (numerical parity with any published result by construction) and is more memory efficient, with interventions expressed through `.trace()`. SAELens now supports SAE inference on nnsight- and HF-loaded models, so this path is viable. The cost is that patching is less turnkey than TL's `act_patch`, and you write more of the intervention plumbing yourself.

**Decision (default): use TransformerLens as the primary instrumentation layer**, for three reasons that all reduce agent error: the patching code is copy-pasteable and well-specified, the SAE hook-point mapping is unambiguous, and the hook-point names are explicit strings rather than module-path guesswork. Mitigate the two costs explicitly. The numerical-drift cost is closed by acceptance test 11.0 (TL logits must match HF-eager within tolerance on a fixed prompt set). The retrieval-head porting cost is closed by Section 5.2 giving the exact algorithm so the agent reimplements on TL hooks rather than guessing.

**Fallback (checkpoint):** if test 11.0 fails to reach tolerance, or if memory becomes blocking, switch the backbone to HF + nnsight and re-run 11.0 (which it passes trivially, being the real model). Surface this to the human before switching; it changes the plumbing in Sections 5 and 8.

---

## 3. Model configuration

### 3.1 Primary and replication models

| Role | Model | Reason |
|---|---|---|
| Primary | `meta-llama/Llama-3.1-8B-Instruct` | The KV-compression and retrieval-head lineages overwhelmingly use Llama; Llama Scope provides SAEs and transcoders on every sublayer of all 32 layers. |
| Replication | `google/gemma-2-9b-it` | **[Errata 2026-06-02]** The original Gemma Scope (Lieberum et al., 2024) provides high-quality SAEs for Gemma 2 but **no transcoders** — transcoders only exist for Gemma 3 (Gemma Scope 2, 2025). The transcoder-based signals are therefore dropped in replication and reported as a finding (§11.2c / §13.6); the attention-based signals show the disagreement structure is not Llama-specific. |
| Third option | Qwen2.5-7B-Instruct | Qwen-Scope (2026) exists; reserve for the cross-model universality extension (Section 10 of the spec). |

### 3.2 Architecture facts that change the code (deterministic)

Llama-3.1-8B uses grouped-query attention: **32 query heads, 8 key/value heads, head_dim 128, 32 layers.** The query-to-KV mapping is `kv_group(q) = q // 4`. This matters in exactly one place and is harmless everywhere else, so state it once and move on:

- Attention-reading signals (accumulated attention, Wu retrieval heads, QRHead) operate at the **query-head** level: there are 32 attention patterns per layer, and `hook_pattern` returns shape `[batch, 32, q_pos, k_pos]` because TL expands the KV heads internally. Retrieval-head and QRHead detection are correct at this granularity.
- KV-cache compression in the flip test (Section 9) uses H2O and SnapKV, which are **token-level** evictions, not head-level. Token-level eviction is unaffected by the query/KV-head distinction. Therefore the GQA granularity issue **does not touch Phase 3** and the agent must not over-engineer for it.
- The granularity mismatch only bites if a *future* extension uses head-level eviction (DuoAttention), in which case query-head retrieval scores must be pooled to KV groups. Note it for the future-work section; do not implement now.

### 3.3 Chat template and the base-SAE-on-instruct decision (checkpoint)

Two tokenization hazards, both named risks for silent index errors.

First, the chat template. Llama-3.1-8B-Instruct renders prompts with special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`). Bricken et al. found that consistent inclusion/exclusion of role tags materially changes probe behavior, so this is not cosmetic. The rule: **always run the full rendered chat template through the model** (the activations must come from the deployment-realistic input), but **compute all token-level importance rankings over the content tokens only**, excluding template/special tokens from the ranking. Build an explicit content-token mask once, from the tokenizer's `apply_chat_template` offsets, and reuse it everywhere. Off-by-one over the template is the canonical bug; acceptance test 11.1e checks the mask by decoding the masked spans.

Second, the base-versus-instruct SAE mismatch (this is a genuine scientific checkpoint, not a coding detail). Llama Scope SAEs were trained on **Llama-3.1-8B-Base** activations, but the deployment-relevant model is **Instruct**. The Llama Scope paper explicitly studied base-to-finetuned transfer and reports reasonable generalization, but the transfer is imperfect. Two options:

- **Recommended:** study Instruct (deployment-realistic, matches the retrieval/QA/compression setting) and apply the base-trained SAEs to it, with acceptance test 11.0c verifying that SAE reconstruction error on Instruct activations is within tolerance (it will be somewhat worse than on base). The captured-variance control (Section 4.4) is what keeps a low SAE-vs-others agreement attributable to genuine plural localization rather than to a lossy dictionary on out-of-distribution activations.
- **Conservative fallback:** study Llama-3.1-8B-Base directly, matching the SAEs perfectly but sacrificing chat realism and the deployment framing. The flip test still works on the base model, but the safety/deployment story weakens.

Surface this to the human. The recommendation is Instruct-plus-base-SAE with the variance control, because the entire payoff of the project is about deployed inference, and the base model is not what gets deployed.

---

## 4. SAE and transcoder specifics

### 4.1 Repositories and the naming convention (deterministic)

Llama Scope checkpoints live at two Hugging Face locations: `fnlp/Llama-Scope` (cited as the checkpoint home in the paper) and `OpenMOSS-Team/Llama-Scope` (the "frontpage" index). Confirm which path SAELens resolves before relying on either. The naming convention is exact and the agent must parse it correctly:

```
L[Layer][Position]-[Expansion]x
  Layer       = 0 .. 31
  Position    = R  (post-MLP residual stream)
                A  (attention output)    <-- DO NOT USE; see 4.2
                M  (MLP output)
                TC (transcoder)
  Expansion   = 8x  (32K features)  |  32x (128K features)

Examples:
  L15R-8x   = layer 15, residual stream, 32K features
  L15TC-8x  = layer 15, transcoder,      32K features
```

There are 256 checkpoints total (32 layers x 4 positions x 2 widths). They are TopK SAEs post-processed to JumpReLU variants, with the decoder-column 2-norm folded into the TopK computation and a K-annealing training schedule. None of those training details affect inference, but the agent should know the activation function is effectively JumpReLU/TopK so the L0 (active-feature count) acceptance check uses the right expected value.

### 4.2 The attention-SAE warning (deterministic, load-bearing)

Per the Llama Scope authors' own README, the attention-output SAEs (`LXA`, whether trained on `z` or `attn_out`) "turn out to have a lot of inactive features" and are explicitly **not recommended**. The agent must not use `LXA` checkpoints for any substrate. This is exactly the kind of fact an agent guessing from a naming pattern would get wrong, and it would quietly poison the component substrate.

### 4.3 Which checkpoints this project uses

For the token-attribution substrate, use **transcoders (`LXTC`)** rather than residual SAEs, because transcoders are built for clean attribution (freeze them and the input-to-feature path is the object you attribute through) and they sidestep part of the irreducible-reconstruction-error objection. Use the 32K-feature (`8x`) width as the default; 128K (`32x`) is a robustness option, not the default, because wider SAEs have higher dead-feature rates and the marginal interpretability is not worth the attribution noise at this stage. Pick a small set of layers to study rather than all 32: a sensible default is three depths, roughly 25%, 50%, and 65% model depth (layers 8, 16, 21), which brackets where prior work finds reasoning and retrieval features concentrate. Surface the layer choice to the human; it is cheap to change.

### 4.4 Transcoder token-attribution recipe (highest-risk component; deterministic recipe, explicit fallback)

This is the single most error-prone component in the project and the one least served by a turnkey published method. The recipe below is defensible; the validation gate and fallback are mandatory, not optional.

**Step 1, find answer-relevant features.** Run the clean forward pass. At the answer position (the last prompt token whose next-token prediction is the answer, or the position of the first answer token), read transcoder feature activations at the chosen layer(s). Score each feature by its effect on the correct-answer logit. Two estimators, in increasing fidelity and cost:
- Direct logit attribution via the logit-lens path: `DLA(f) = act_f * (W_dec[f] . W_U[answer_token])`, where `W_dec[f]` is the transcoder decoder vector for feature `f` written into the residual stream and `W_U` is the unembedding. Cheap, ignores downstream nonlinearity.
- Gradient of the answer logit with respect to `act_f`, times `act_f`. One backward pass; captures downstream paths approximately.

Take the top `M` features by `|score|`, with `M` in the range 16 to 64.

**Step 2, attribute those features back to context tokens.** The feature activation at the answer position is a function of context tokens through attention in layers up to the transcoder's layer. For each top feature `f`, compute the gradient of `act_f` with respect to the residual stream at each context token position, contract over the residual dimension, and (optionally) multiply by the clean activation for a gradient-times-activation attribution, or by the corruption delta for an attribution-patching estimate. Weight each feature's per-token attribution by that feature's Step-1 answer-importance, and sum over the `M` features to get the per-instance token-importance vector for the transcoder substrate. Cost is `O(M)` backward passes per instance, acceptable for modest `M`.

**Mandatory validation gate (test 11.2c).** On instances the model answers correctly, the transcoder token ranking must (a) place gold-span tokens above random context tokens (AUROC versus gold-span membership above chance), and (b) correlate at least weakly with the ablation oracle of Section 8 (Spearman above a modest bar, e.g., 0.3).

**Fallback if the gate fails.** Drop the transcoder from the **token-attribution** substrate and retain it only in the **component** substrate (via attention-to-feature read-direction loadings). Report the failure as a finding: "transcoder-based token attribution does not reliably localize on long-context QA in this setting." That is a legitimate, publishable negative about a method many people assume works, and it costs you nothing in the overall design because the token-attribution substrate still has the two attention-based members plus the gold-span oracle.

---

## 5. The signals: exact algorithms

Each token-substrate signal must emit, per instance, a real-valued importance vector over the content tokens, normalized per instance (min-max or z-score; pick one and apply uniformly). Each component-substrate signal emits a score over heads.

### 5.1 Accumulated-attention importance (H2O / Scissorhands operationalization)

Define `importance(t) = mean over selected (layer, head) of sum over query positions q in Q of A[layer, head, q, t]`, where `A` is the eager attention pattern and `Q` is a query region.

- **Primary variant:** `Q` = the question tokens plus the generated answer tokens (attention paid *by the act of answering* onto each context token). This is the most faithful operationalization of "what did the model attend to when producing the answer."
- **Robustness variant:** `Q` = all positions (the cumulative heavy-hitter definition closest to H2O/Scissorhands as published). Report as a robustness column, not the headline.

Aggregate across heads and layers by mean. Two attention-based signals (this and retrieval heads) both read attention, so their mutual agreement is partly mechanical and is *not* the interesting comparison; the informative comparison is attention-family versus transcoder versus oracle.

### 5.2 Wu retrieval heads (exact decoding-time algorithm; port, do not pip-install)

Reference: `github.com/nightdessert/Retrieval_Head`, `retrieval_head_detection.py`. The precise scoring rule, stated cleanly:

```
Construct a needle-in-haystack instance: a needle containing answer string k,
inserted at position p inside haystack c. Prompt the model to retrieve k.

Decode the answer autoregressively. At each decoding step where the model
generates a token g that is part of the needle answer k:
    for each query head h:
        let j = argmax over key positions of A[layer, h, current_pos, :]
        if token at key position j == g  AND  position j lies inside the needle span:
            head h is "copy-pasting" at this step -> increment its hit count

retrieval_score(h) = (number of copy-paste hits for h) / (total needle tokens generated),
                     averaged over many (needle, position) samples.
```

Implementation notes that prevent silent errors: read attention with eager attention (Section 1.2); score only steps where the generated token is in the needle (not all steps); the "inside the needle span" check requires tracking the needle's token indices in the full sequence (re-use the content-token mask machinery). Output format, matching the reference for cross-checkability: `head_score/*.json` as `{ "layer-head_id": [list of per-sample scores] }`. Sample budget: the authors note that few samples stably surface the strongest heads; use 20 to 50 (needle, position) pairs spread across positions. A single 80GB GPU handles up to 50K context, but detection at 4K to 8K is sufficient and faster; set the context length low if iterating.

### 5.3 QRHead, and why both are in the study (turning a fork into a result)

There is a live scientific fork the scientific spec glossed: Wu's copy-paste retrieval heads are **not** the only claimant to the label "retrieval head." QRHead (Query-Focused Retrieval Head, EMNLP 2025, arXiv 2506.09944) identifies heads by aggregating attention with respect to the input query using a handful of *real-task* examples rather than synthetic NIAH copy-paste, and on Llama-3.1-8B, masking the top-32 QRHeads degrades NIAH *more* than masking the top-32 Wu heads. Separately, "Retrieval Heads are Dynamic" (arXiv 2602.11162) argues the set should be conditioned on context rather than fixed.

**Decision: include Wu retrieval heads and QRHead as two distinct component-substrate signals, and report their mutual agreement as one of the headline disagreement numbers.** This is the right call precisely because the project's thesis is that methods claiming to localize the same thing disagree. Wu-versus-QRHead is a clean, concrete instance of two "retrieval head" detectors that demonstrably differ, identified by the QRHead authors themselves. It costs little extra to compute and it strengthens the paper rather than muddying it. QRHead algorithm: for a small set of real long-context QA examples, accumulate each head's attention mass onto the query-relevant tokens, rank heads by that accumulated mass, take the top set (match the count to the Wu set for a fair Jaccard).

### 5.4 ITI head selection (component substrate only)

The internal-confidence methods do not belong in the token-attribution substrate (they read the answer position, not the context). ITI does contribute a *head set* to the component substrate. Recipe: train a linear probe on each head's per-token output activations (`hook_z`, head-sliced) to classify truthful versus untruthful on a labeled true/false dataset; rank heads by held-out probe AUROC; take the top-K, with K matched to the retrieval-head count for a fair Jaccard. Use a labeled factual true/false set for probing (the ITI setup uses TruthfulQA with cross-validation folds; to avoid using an eval set as training data, a separate true/false statement set is cleaner). This requires a probing dataset distinct from the QA evaluation sets in Section 10.

### 5.5 Orgad exact-answer-token signal (RQ4, kept separate)

This signal answers RQ4 (attribution versus internal confidence) and is reported separately from the token-attribution matrix. Identify exact-answer tokens by locating the answer string within the generated tokens for extractive cases; for free-form answers, Orgad's method uses heuristics plus an instruction-tuned LLM to mark the exact-answer tokens. Implement the extractive version first (deterministic, cheap), and treat the LLM-assisted extension as optional. The signal is then the truth-probe read at those answer-position tokens, compared geographically against where the token-attribution methods place importance.

---

## 6. Substrate definitions and projections (deterministic)

Never report a single agreement number across incommensurable objects. Agreement is defined only within a substrate, with the projection stated explicitly.

```
TOKEN-ATTRIBUTION SUBSTRATE  (importance over content tokens)
  members:
    - accumulated attention            (native, Section 5.1)
    - Wu retrieval-head attention       (sum top-Wu-heads' attention onto each token)
    - QRHead attention                  (sum top-QR-heads' attention onto each token)
    - transcoder attribution            (Section 4.4, IF it passes gate 11.2c)
  oracle reference:
    - gold answer span (where annotated)
    - token-ablation causal effect (Section 8)

COMPONENT SUBSTRATE  (score over query heads)
  members:
    - Wu retrieval heads                (native)
    - QRHead                            (native)
    - ITI truth heads                   (native, Section 5.4)
    - transcoder attention-to-feature loadings  (read-direction overlap)
  excluded: accumulated attention (token-level only)

ANSWER-POSITION SUBSTRATE  (RQ4, reported separately)
  members:
    - ITI truth direction read at answer position
    - Orgad exact-answer-token probe
  compared geographically against token-attribution placement, NOT merged into it
```

Carry forward the warning the spec raised: the two attention-based members of the token substrate will agree partly because they both read attention. State this in the writeup and foreground the cross-family comparisons (attention vs transcoder vs oracle) as the load-bearing ones.

---

## 7. Agreement metrics (deterministic)

Per substrate, per instance, compute the following over the relevant ranking vectors.

- **Spearman rank correlation** between each pair of method importance vectors; report the mean over instances with a bootstrap 95% CI, as a pairwise heatmap.
- **Top-k Jaccard overlap** for k in {gold-span size, 5, 10, 20}, and **rank-biased overlap (RBO)** for a top-weighted, length-robust view.
- **Agreement with the gold span** (token substrate, where annotated): precision@k and AUROC of each method's importance against gold-span membership.
- **Per-instance disagreement score**, the quantity carried into the flip test:

```
D(x) = 1 - mean over method pairs (i,j) of Spearman( imp_i(x), imp_j(x) )
```

- **Permutation test** for whether cross-family agreement exceeds chance: shuffle token positions within instance, recompute pairwise agreement, build the null.
- **Multiple-comparison control:** across datasets and budgets, apply Benjamini-Hochberg FDR to the reported p-values.

---

## 8. The causal oracle (deterministic, with the corruption decision called out)

The oracle is what separates "the tools are noisy" (outcome 1) from "localization is genuinely plural" (outcome 2). Get this right or the central finding is unsupported.

### 8.1 Ground-truth effect: token ablation aligned with the deployment operation

Define the per-token causal effect as the change in the correct-answer log-probability when token `t` is removed from the KV cache:

```
E(t) = logP(answer | full context) - logP(answer | context with token t masked from all attention)
```

This is an **ablation**, and it is deliberately the *same operation* as the downstream failure mode, because KV eviction is exactly "remove tokens from the cache." That alignment is a feature: the oracle measures the thing the flip test stresses. The cost is `O(context length)` forward passes per instance (one per ablated token), which is why Section 8.3 approximates it.

The scientific spec used "ablate" and "patch" interchangeably; they are different and the distinction matters. Ablation removes a token; patching swaps a token's activation from a corrupted run. Use ablation for `E(t)` because of the deployment alignment above.

### 8.2 Corruption choice for any patching-based estimate (checkpoint, with a default)

Where patching is used (the attribution-patching approximation in 8.3 needs a corrupted reference for the activation delta), the corruption must be chosen deliberately. Zhang and Nanda (arXiv 2309.16042) show that Gaussian-noise corruption (the ROME style) is fragile and hyperparameter-sensitive. **Default: symmetric token replacement** (swap the gold span, or the needle, for a different plausible span so the answer is no longer supported), which is more stable and semantically controlled than noising embeddings. Document-swap (replace the supporting document with a distractor) is a coarser alternative for the multi-hop sets. Do not use Gaussian noise as the primary corruption. Report top-token-set stability across two corruption variants (test 11.2b) so a reviewer cannot claim the oracle is as arbitrary as the methods it judges.

### 8.3 Cheap approximation: attribution patching, validated against ablation

Attribution patching (Syed et al.) approximates a patching/ablation effect with one forward and one backward pass using a first-order expansion. For the mean-ablation variant aligned with 8.1:

```
E_hat(t) ~= ( a_mean(t) - a_clean(t) )^T  *  grad_{a(t)} [ logP(answer) ]
```

evaluated with the clean-run gradient, where `a(t)` is token `t`'s contribution (residual or KV) and `a_mean(t)` is the mean-ablation reference. This collapses the per-token forward-pass sweep into a single backward pass.

**Mandatory validation gate (test 11.2a):** on a 5 to 10 percent held-out subsample, compute true `E(t)` (the expensive ablation) and require **Spearman(`E_hat`, `E`) >= 0.8** on token ranking, with high sign-agreement. We require Spearman on ranking, not Pearson on magnitude, because every downstream use ranks tokens. If the bar is not met, fall back to subsampled true ablation (fewer instances, exact effect) rather than trusting the approximation.

### 8.4 Cost mitigations (deterministic)

Run the oracle-validated core at 1K to 4K context (the central finding does not require 40K). Compute `E` at **sentence or span granularity** rather than per token wherever the gold annotation is sentence-level (HotpotQA supporting facts are sentences), which cuts the sweep by an order of magnitude. Use attribution patching (8.3) for the full sweep with the subsample verification. These three together keep Phase 2 on a single GPU.

### 8.5 Method-versus-oracle fidelity and the outcome decision

Per instance, correlate each method's importance with `E`. The outcome of the whole project is decided here, not by agreement alone:

- methods disagree with each other **and** all track `E` poorly: outcome 1, tools are noisy (weakest).
- each method tracks `E` decently **but** they disagree with each other: outcome 2, multiple causally-real localizations, the strong finding MIB's aggregate framing cannot produce.
- one method tracks `E`, others do not: outcome 3, a ranking of deployed methods on real QA (Kantamneni-style verdict on objects Kantamneni never tested).

---

## 9. The compression / flip test (the payoff)

### 9.1 Compression methods and the reproduce-the-paper gate

Use **H2O** and **SnapKV** as the two eviction methods, swept over budget. Reimplementing eviction by hand is precisely where silent bugs live, so use an existing, maintained KV-compression library that exposes these methods under a unified Hugging Face generate interface (NVIDIA's KVPress is the obvious candidate and implements H2O, SnapKV, StreamingLLM and others as composable presses; verify it is current and supports the target model before committing). The agent must not trust the library blindly: acceptance test 11.3a requires that the chosen method, at a stated budget, roughly reproduces its paper-reported LongBench numbers (within a few points). That test holds regardless of which library is used and catches a misconfigured eviction that would otherwise corrupt every Phase 3 result.

### 9.2 Correctness metric (checkpoint, pre-register the primary)

"Correct" must be defined to label flips, and the definition changes the headline numbers, so it is pre-registered, not chosen post hoc.

- **Synthetic NIAH:** exact match on the inserted needle string (unambiguous).
- **Natural QA (HotpotQA, LongBench QA subsets):** SQuAD-style normalization (lowercase, strip articles and punctuation), then **primary metric F1 >= 0.5 counts as correct**, with **exact match reported as a robustness column**. Use LongBench's official per-task metric where the subset defines one.
- Keep the core deterministic: do **not** use an LLM judge for the primary flip labeling (cost, nondeterminism, and a hidden dependency). An LLM-judge pass is an optional robustness check only.

Report sensitivity of the Phase 3 conclusion to the F1 threshold (0.3, 0.5, exact match), because the scientific spec correctly flagged that this choice moves the numbers.

### 9.3 The flip-prediction model (deterministic) and leakage prevention

```
1. Select instances answered CORRECTLY under the FULL (uncompressed) cache.
2. Apply H2O and SnapKV at a fixed budget (sweep budget). Label flip = correct -> incorrect.
3. Fit  logistic:  flip ~ D(x)                              (D from Section 7)
4. Fit  logistic:  flip ~ D(x) + length + gold_depth + answer_entropy   (confound model)
5. Report  AUROC under 5-fold CV  for BOTH models,
   the incremental AUROC of adding D(x) over the confound-only model,
   and a likelihood-ratio test (chi-square) for the D(x) term.
6. Report calibration of the D(x) model.
```

**The claim is only interesting if D(x) adds signal beyond trivial difficulty proxies** (length, needle/supporting-fact depth, answer-token entropy as a model-confidence proxy). If D(x) does not beat the confound-only model, you have rediscovered that hard questions break under compression, which is not the finding. Report the confound-only AUROC *before* adding D(x) so the increment is interpretable.

**Leakage prevention (named bug):** D(x) must be computed strictly from the full-cache forward pass, with no information from the compressed run. The agent must structure the pipeline so the compressed run cannot influence any feature used to predict its own outcome. Sample size: aim for at least 300 correctly-answered instances per budget so there are enough flips to fit the model; report the base flip rate (acceptance test 11.3b requires it to be neither ~0% nor ~100%, by choice of budget).

---

## 10. Datasets (deterministic identifiers, fields, sample sizes)

| Purpose | HF identifier / source | Config | Gold-span source | Phase-1 n | Oracle n | Flip n |
|---|---|---|---|---|---|---|
| Controllable NIAH | `gkamradt/LLMTest_NeedleInAHaystack` methodology (Paul Graham haystack + synthetic "magic number" needles); or RULER | n/a | exact by construction (you insert it) | 200-500 | 100-200 | >=300 correct |
| Multi-hop QA | `hotpotqa/hotpot_qa` | `distractor` | `supporting_facts` (sentence-level) | 200-500 | 100-200 | >=300 correct |
| Long-context QA | `zai-org/LongBench` | `hotpotqa`, `2wikimqa`, `musique` | per-subset gold context (answer strings) | 200-500 | subset | subset |
| Easy single-hop contrast | `mandarjoshi/trivia_qa` | `rc` | answer string + aliases | 200 | 50-100 | n/a |

**[Errata 2026-06-02]** The three HF identifiers above were corrected after verifying schemas against the live datasets-server. All three originals were renamed and now 404 with `{"error":"The dataset has been renamed."}`: `hotpot_qa` -> `hotpotqa/hotpot_qa`, `THUDM/LongBench` -> `zai-org/LongBench`, `trivia_qa` -> `mandarjoshi/trivia_qa`. `zai-org/LongBench` is script-based (`LongBench.py` + `data.zip`), so loading it requires `trust_remote_code=True`; it has no datasets-server introspection. Verified columnar shapes: HotpotQA `supporting_facts`/`context` are dicts of parallel lists; TriviaQA `entity_pages`/`search_results` are columnar (`entity_pages.wiki_context` is a `list[str]`) while `answer` is a plain dict; LongBench QA rows are flat (`input`, `context`, `answers: list[str]`). These IDs are pinned as module constants in `lcv.data.{hotpotqa,longbench,triviaqa}`.

**Gold-span token mapping (named bug surface).** Mapping HotpotQA `supporting_facts` (given as `[title, sentence_id]`) to token indices requires: locating the sentence text in the rendered context, then mapping characters to tokens through the tokenizer's offset mapping, then intersecting with the content-token mask. Acceptance test 11.1e: decode the mapped token indices and fuzzy-match the result to the gold sentence text at >= 0.9 similarity. For synthetic NIAH this is trivial and exact (you control insertion), which is why NIAH is the cleanest substrate to bring up first.

Use NIAH for the easy/hard contrast as well by varying needle depth and haystack length; TriviaQA gives a natural-text easy contrast against the multi-hop sets.

---

## 11. Per-phase acceptance-test checklist

This is the operational heart of the addendum. The agent runs these before advancing; the human signs off at each boundary. A green phase means "the instrumentation is provably doing what we think," not "it ran."

### 11.0 Phase 0: environment and parity

- **11.0a** Greedy generation on five sanity prompts is coherent (manual human read).
- **11.0b** *(TL path)* TransformerLens logits match HF-eager logits on a fixed 10-prompt set within tolerance: next-token KL < 1e-3, or max abs logit difference within a documented bound after TL processing. Fail -> switch to the nnsight fallback (Section 2) and re-run.
- **11.0c** Each SAE/transcoder loads at its named hook point; reconstruction error (1 minus explained variance) on a sample is within the layer's published Llama Scope range; feature activations are sparse with L0 matching the TopK/JumpReLU expectation. Fail -> wrong hook point or wrong checkpoint.
- **11.0d** Attention extraction returns shape `[batch, 32, q, k]` and every attention row sums to 1.0 within fp tolerance. Fail -> FlashAttention returning None, or a softmax-axis error.

### 11.1 Phase 1: signals are correct

- **11.1a** Wu retrieval-head detection reproduces the published Llama-3.1-8B retrieval heads: a substantial fraction of the detected top-20 overlaps the released `head_score` set (quantify and pre-register the bar). Fail -> the decoding-time scoring rule is wrong.
- **11.1b** Ablating the top-N Wu retrieval heads collapses NIAH accuracy, while ablating N random heads barely changes it. This reproduces Wu's causal result and validates both detection and the ablation machinery in one test.
- **11.1c** Masking the top-32 QRHeads degrades NIAH at least as much as masking the top-32 Wu heads (reproduces the QRHead headline on Llama-3.1-8B).
- **11.1d** Accumulated-attention importance on a needle task ranks needle tokens above random context tokens (AUROC vs needle membership > chance). Fail -> attention-aggregation bug.
- **11.1e** Gold-span token mapping: decoded mapped spans fuzzy-match gold sentence text >= 0.9; content-token mask, when decoded, excludes all chat-template special tokens. Fail -> off-by-one over the template or a tokenizer-offset error.

### 11.2 Phase 2: the oracle

- **11.2a** Attribution patching vs true ablation on the held-out subsample: Spearman >= 0.8 on token ranking with high sign-agreement. Fail -> fall back to subsampled true ablation.
- **11.2b** Top-token-set stability across two corruption variants (token-swap vs document-swap): report the overlap. Not a hard gate; a reported robustness number that pre-empts the "arbitrary oracle" objection.
- **11.2c** Transcoder token-attribution gate (Section 4.4): gold-span AUROC > chance and Spearman with oracle > ~0.3 on correctly-answered instances. Fail -> drop transcoder from the token substrate, keep in component substrate, report as a finding.
- **11.2d** Sanity: ablating the entire gold span produces a large answer-logit drop on instances the model got right (the gold span is causally necessary). Fail -> the answer-metric wiring is wrong.

### 11.3 Phase 3: the payoff

- **11.3a** The compression method, at a stated budget, roughly reproduces its paper-reported LongBench numbers (within a few points). Fail -> misconfigured eviction; do not proceed.
- **11.3b** Base flip rate is in a usable band (roughly 5% to 40%), tuned by budget. 0% or ~100% -> adjust budget; there is no signal at the extremes.
- **11.3c** Confound-only AUROC is computed and reported *before* D(x) is added, and D(x) is provably computed from the full-cache run only (leakage check passes).

### 11.4 Phase 4: replication

- Re-run 11.0 through 11.3 on `gemma-2-9b-it` with the **original Gemma Scope** (SAE-only). **[Errata 2026-06-02]** Gemma 2 has no transcoder suite (transcoders are Gemma-3/Gemma Scope 2 only), so the transcoder-based signals are dropped in replication and reported as a finding (§11.2c / §13.6); the attention-based signals carry RQ1–RQ3. The disagreement structure and the D(x)-predicts-flips result either replicate or they do not; either is reportable.

---

## 12. Repository layout and module breakdown

Build modularly so the human can validate at module boundaries that line up with the acceptance tests. Suggested structure:

```
repo/
  env/                 uv lockfile, pinned versions, model + SAE download scripts
  model/
    backbone.py        TL HookedTransformer load (+ nnsight fallback), parity test 11.0b
    chat_template.py   render + content-token mask (test 11.1e)
    sae_loader.py      Llama Scope / Gemma Scope load by name; reconstruction check 11.0c
  signals/
    attention_hh.py    accumulated-attention importance (5.1)
    retrieval_wu.py    Wu retrieval-head detection (5.2), reproduces 11.1a/b
    retrieval_qr.py    QRHead detection (5.3), reproduces 11.1c
    transcoder_attr.py transcoder token-attribution (4.4), gate 11.2c
    iti_heads.py       ITI head selection (5.4)
    orgad_tokens.py    exact-answer-token signal (5.5)
  substrates.py        projections + normalization (Section 6)
  agreement.py         Spearman/Jaccard/RBO/D(x)/permutation (Section 7)
  oracle/
    ablation.py        true token/span ablation E(t) (8.1)
    attr_patching.py   attribution-patching approximation + gate 11.2a (8.3)
    corruption.py      token-swap / document-swap (8.2)
  compression/
    flip_test.py       eviction via external lib + reproduce-paper gate 11.3a (9.1)
    correctness.py     EM/F1/normalization, pre-registered metric (9.2)
    flip_model.py      logistic regression + confounds + CV + LRT (9.3)
  data/
    niah.py            synthetic needle generation (exact gold spans)
    tokenization.py    content-token mask + char-span -> token mapping (3.3, 11.1e)
    assembly.py        shared natural-text Instance assembly + gold mapping (11.1e)
    hotpotqa.py        loader + supporting-fact -> token mapping (11.1e)
    longbench.py       subset loaders + official metrics
    triviaqa.py        easy-contrast loader
  tests/
    phase0.py ... phase4.py   the Section 11 gates, runnable
  analysis/
    figures.py         heatmaps, fidelity distributions, flip ROC
```

The interfaces that matter for human review are the signal outputs (every token-substrate signal returns a normalized per-token vector with the same shape and masking; every component signal returns a per-head score) and the oracle output (per-token `E`). If those contracts hold, the agent's downstream statistics are mechanical and low-risk.

---

## 13. What the agent must not decide alone (checkpoints, collected)

Surface each of these to the human; do not resolve autonomously.

1. **Instrumentation backbone** if parity test 11.0b fails (TL vs nnsight fallback). Section 2.
2. **Base-SAE-on-Instruct vs study-the-base-model.** Section 3.3. Default recommended: Instruct + base SAE + variance control.
3. **Which layers** to instrument for the transcoder substrate. Section 4.3. Default: ~25/50/65% depth.
4. **Corruption type** for patching. Section 8.2. Default: token-swap; never Gaussian noise as primary.
5. **Correctness metric and threshold** for the flip test. Section 9.2. Pre-registered default: F1 >= 0.5 primary, EM robustness.
6. **Dropping the transcoder token substrate** if gate 11.2c fails. Section 4.4. This is a finding, not a silent fallback; the human should confirm it is reported as such.
7. **Budget points** for the flip test, tuned so 11.3b lands in the usable band. Section 9.3.

Everything else in this document is deterministic and the agent should implement it as written, verifying external-library specifics against the reproduce-a-published-number acceptance tests rather than against my naming.

---

## 14. Decisions log and known limitations

Recorded so future-you remembers why, and so reviewers see the choices were deliberate.

- **TransformerLens primary** for unambiguous hook points and turnkey patching, at the cost of HF numerical parity, which is closed by test 11.0b. nnsight is the parity-exact fallback.
- **Instruct model with base-trained SAEs** for deployment realism, accepting somewhat higher SAE reconstruction error (Llama Scope reports acceptable base-to-finetuned transfer), with the captured-variance control guarding the interpretation.
- **Transcoders over residual SAEs** for cleaner attribution; attention-output SAEs excluded entirely on the authors' own dead-feature warning.
- **Wu and QRHead both included** as distinct signals, converting a scientific fork into one of the headline disagreement measurements; this also pre-empts the "you used the wrong retrieval-head definition" review objection.
- **Token ablation as the oracle**, deliberately the same operation as KV eviction, so the oracle measures the thing the payoff stresses; attribution patching is the validated cheap approximation.
- **Token-level eviction (H2O, SnapKV)** for the flip test, which sidesteps the GQA query/KV-head granularity issue entirely; head-level eviction (DuoAttention) is deferred to future work where the query-to-KV-group pooling becomes necessary.
- **Deterministic correctness metric** (no LLM judge in the core) for reproducibility; judge is an optional robustness pass.
- **Core at 1K to 4K context** with sentence-granularity oracle and attribution-patching, keeping the whole project on a single 80GB GPU.

Known limitations to state in the paper: SAE/transcoder attribution noise (mitigated and gated, but real); base-to-instruct SAE transfer imperfection; retrieval-head definitions are contested and possibly dynamic (which this study treats as signal, not noise); the oracle's corruption choice is one of several defensible options (robustness reported); and findings are established on two model families, with cross-model universality left as the named extension.
