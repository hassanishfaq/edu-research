# Methodology

This document gives the formal description of the framework implemented by the prototype. It explains how each equation maps to code, and the rationale behind the main design decisions. Equation numbers in parentheses refer to Chapter 4 of the thesis.

## Contents

1. [Problem formulation](#1-problem-formulation)
2. [Multi-antivirus label service](#2-multi-antivirus-label-service)
3. [Normaliser: from nested report to schema row](#3-normaliser-from-nested-report-to-schema-row)
4. [Serialisation and the sequence-length policy](#4-serialisation-and-the-sequence-length-policy)
5. [Encoder slots](#5-encoder-slots)
6. [Stacking](#6-stacking)
7. [Unknown-category rule](#7-unknown-category-rule)
8. [Training protocol](#8-training-protocol)
9. [Category hold-out protocol](#9-category-hold-out-protocol)
10. [Computational cost of the transformer](#10-computational-cost-of-the-transformer)
11. [Design rationale](#11-design-rationale)
12. [References](#12-references)

---

## 1. Problem formulation

Let $\mathcal{F}$ be a set of files and $\mathcal{E} = \{e_1, e_2, e_3\}$ a set of antivirus engines. Here the engines are Baidu, Kaspersky and Symantec, the three engines present in the report corpus. For a file $f$:

- $A(f)$ is its **behaviour report**, the nested JSON document returned by a sandbox (Eq. 4.5);
- $r_e(f)$ is the **vendor string** of engine $e$, for example `W32.Sality.AE`, or $\varnothing$ when the engine does not flag the file.

Let $\mathcal{K} = \{\kappa_1, \dots, \kappa_9\}$ be the **standardised categories** of thesis Table 4.3: Adware, Backdoor Trojans, Worms, Downloaders, PUA, Infostealers, Trojans, Viruses and Fake Security Tools.

The task is to learn $g: A(f) \mapsto \hat\kappa \in \mathcal{K}$ from reports labelled by an engine consensus. At inference time only the report is needed. In addition, $g$ must report *unknown category* for a report whose category is not represented in training.

## 2. Multi-antivirus label service

### 2.1 Category map

Each engine has a naming convention. Baidu uses `<platform>.<type>.<family>.<variant>`, Kaspersky `[not-a-virus:]<behaviour>.<platform>.<family>.<variant>`, and Symantec `<type>.<family>[!suffix]` or `W32.<family>`. A **versioned category map** (`category_map.yaml`) holds, per engine, an ordered list of regular-expression rules:

$$
\mu_e(r) = \kappa_j \ \text{ for the first rule } (\rho_j, \kappa_j) \text{ with } \rho_j \text{ matching } r, \qquad \mu_e(r) = \textsf{Unmapped} \ \text{ if no rule matches.}
$$

Version 1 contains 28 rules that follow Table 4.3. For example, Kaspersky `Lollipop` and `Linkun` adware is filed under PUA, and Symantec `Trojan.FakeAV` under Fake Security Tools. Rootkit strings are deliberately left unmapped, so the map-gap workflow can be seen on the dashboard.

### 2.2 Aggregated score and consensus

The aggregated detection score of Eq. 4.4 counts the engines that flag a file:

$$
\mathrm{Score}(f) = \sum_{e \in \mathcal{E}} \mathbb{1}\big[r_e(f) \neq \varnothing\big] \in \{0, 1, 2, 3\}.
$$

The **consensus category** is the majority over the mapped verdicts, ignoring `Unmapped`:

$$
\kappa^{\ast}(f) = \arg\max_{\kappa \in \mathcal{K}} \sum_{e} \mathbb{1}\big[\mu_e(r_e(f)) = \kappa\big].
$$

Ties are broken by an engine priority list (`labels.engine_priority`). A file enters the labelled corpus when $\mathrm{Score}(f) \ge s_{\min}$ (`labels.min_engines`, default 1) and $\kappa^{\ast}(f) \neq \textsf{Unmapped}$. A file on whose mapped categories the engines disagree is marked **disputed**. The share of multi-engine files without dispute is the **engine-agreement rate**, which serves as a drift signal (Section 4.4.4 of the thesis).

Every verdict stores the raw vendor strings and the map version. When the map changes, labels can therefore be re-derived retroactively from the stored strings.

## 3. Normaliser: from nested report to schema row

**Flattening (Eq. 4.6).** The report is flattened to dotted keys. Lists of objects fan out into their fields, so that, for example, `filesystem.opened[*].path` becomes one key with all opened paths:

$$
F_{\text{flat}}(A(f)) = \{\, k_1 \mapsto [v_{1,1}, v_{1,2}, \dots],\ k_2 \mapsto [\dots],\ \dots \,\}.
$$

**Filtering (Eq. 4.7).** Unnecessary keys $K_{\text{unnecessary}}$ are removed. These are the per-event `success` flags (`normalizer.drop_keys`). Two value transformations are applied as part of the same step:

- the sample's own 64-hex-digit SHA-256 name, which is unique per file and carries no behaviour, becomes `<sample>`;
- system and user path prefixes are canonicalised: `C:\WINDOWS\system32` becomes `%SYS32%`, `C:\Documents and Settings\<user>` becomes `%USERPROFILE%`, and so on. The same behaviour then produces the same tokens on different sandbox images.

**Alignment and padding (Eqs. 4.8–4.9).** The key set $C$ is the **schema registry**. Version `v1` has the 39 behaviour columns of the thesis CSV (`sigstack/flatten.py: SCHEMA_V1`). For each $c \in C$:

$$
AV_{\text{final}}(f, c) = \begin{cases} \text{join}\big(\text{distinct}(v_{c,1}, \dots, v_{c,m})\big) & c \in \mathrm{Keys}(AV_{\text{filtered}}(f)) \\ \texttt{"0"} & \text{otherwise,} \end{cases}
$$

where at most `normalizer.max_items_per_key` (20) distinct values are kept per key, in order of appearance. The rows over the corpus form the matrix $M$ of Eq. 4.10. Keys outside the schema are reported, not silently dropped, so a sandbox upgrade that introduces new keys becomes visible.

## 4. Serialisation and the sequence-length policy

The row is serialised in a **fixed category order**: hooking (HK), process (PR), network (NT), registry (RG), file system (FS), mutex (MX), service (SV), runtime DLLs (DL), windows (WN), extra (EX), hosts file (HF):

```
<s> HK: method SetWindowsHook; type WH_MSGFILTER, WH_CBT </s> PR: created proc qppk….exe, … </s> NT: udp 64.4.10.33:123 </s> RG: none </s> FS: … </s>
```

The dashboard displays `<s>`/`</s>` as the thesis's `[CLS]`/`[SEP]`. The encoder reads at most $n = 512$ RoBERTa tokens. Because the order is fixed, truncation always removes the **same low-priority tail**, and the dashboard shows which segments survive: *full*, *partial* or *dropped*.

On the corpus of the thesis, the median serialised report has 710 tokens and 55.8 % of reports exceed 512 tokens. The order therefore places the categories whose events are most discriminative and least voluminous (hooks, processes, network, registry) before the voluminous file-system and DLL lists. The cache keeps up to 3,000 characters per report, which is more than 512 tokens at about 2.5 characters per token.

## 5. Encoder slots

The prototype has four **model slots** with a common interface (`fit`, `predict_proba`, `payload`, `info`):

| Slot | Target | Classes (thesis corpus) |
|---|---|---|
| `category` | consensus category $\kappa^{\ast}$ | 9 |
| `Baidu` | Baidu family string $r_{\text{Baidu}}$ | 80 (families with ≥ 30 labelled reports) |
| `Kaspersky` | Kaspersky family string | 75 |
| `Symantec` | Symantec family string | 72 |

### 5.1 RoBERTa slot (the thesis model)

A fine-tuned `roberta-base` sequence classifier (12 layers, $d = 768$, 12 heads, 124.6 M parameters) reads the serialised report truncated to $n = 512$ tokens:

$$
p_s(\cdot \mid f) = \operatorname{softmax}\big(W_s\, h_{\texttt{[CLS]}}(\text{text}(f)) + b_s\big).
$$

The model is fine-tuned with batch size 8, 500 warm-up steps and weight decay 0.01, with stratified five-fold cross-validation (thesis Section 4.3, `scripts/colab_finetune_roberta.py`). Serving optionally applies dynamic INT8 quantisation to the linear layers for CPU hosts (`encoder.roberta.int8_dynamic`).

### 5.2 Stand-in slot (CPU)

Until fine-tuned folders are available, all slots share one unsupervised feature space built over the corpus, which plays the frozen encoder:

$$
\phi(\text{text}) = \frac{U^\top\, \mathrm{tfidf}\big(\mathrm{hash}_{2^{17}}(\text{1–2-grams of text})\big)}{\big\lVert U^\top\, \mathrm{tfidf}(\cdot) \big\rVert_2} \in \mathbb{R}^{256},
$$

where $U$ holds the first 256 components of a truncated SVD. On the thesis corpus these components explain about 59 % of the variance. Each slot trains its own **logistic-regression head** on $\phi$ (`encoder.surrogate.head: logreg | mlp`). This is the TF-IDF/linear baseline family of thesis Table 4.12, and it is labelled as a stand-in throughout the dashboard.

### 5.3 Folding family models into categories

An engine's family model is mapped into the category space through that engine's category map:

$$
q_e(\kappa \mid f) = \sum_{\phi\,:\ \mu_e(\phi) = \kappa} p_e(\phi \mid f),
$$

where the sum runs over the families $\phi$ of engine $e$. The direct slot already predicts categories: $q_{\text{cat}} = p_{\text{cat}}$.

## 6. Stacking

The meta-learner input concatenates the four category-level vectors:

$$
z(f) = \big[\, q_{\text{cat}}(f) \;\|\; q_{\text{Baidu}}(f) \;\|\; q_{\text{Kaspersky}}(f) \;\|\; q_{\text{Symantec}}(f) \,\big] \in \mathbb{R}^{4 \times 9 = 36},
$$

and a multinomial **logistic regression** (thesis Table 4.11) predicts the category:

$$
\hat p(\kappa \mid f) = \operatorname{softmax}\big(W z(f) + b\big)_\kappa, \qquad W \in \mathbb{R}^{9 \times 36}\ (333 \text{ parameters}).
$$

The meta-learner learns *where each engine's view is reliable*. For example, it learns which engine is the best source for Worms and which is unreliable on PUA. It does so without the best engine having to be known in advance.

### Support-weighted Stacking Precision and F1 (thesis Table 4.10)

For comparison with the thesis, the prototype also computes the support-weighted aggregation of the per-engine metrics:

$$
\mathrm{SP}(\kappa) = \frac{\sum_{e} P_e(\kappa)\, S_e(\kappa)}{\sum_e S_e(\kappa)}, \qquad
\mathrm{SF_1}(\kappa) = \frac{\sum_{e} F_{1,e}(\kappa)\, S_e(\kappa)}{\sum_e S_e(\kappa)},
$$

where $P_e$, $F_{1,e}$ and $S_e$ are the precision, F1 and support of engine $e$'s model on its own labels. An engine that never labels a category contributes nothing to it.

## 7. Unknown-category rule

The unknown-category threshold (Eq. 4.11) is the $\beta$-quantile of the meta-learner's **out-of-fold** confidences on known data:

$$
\tau_4 = Q_\beta\Big\{\, \max_{\kappa}\hat p(\kappa \mid f) : f \in V_{\text{known}} \,\Big\}, \qquad \text{report "unknown category" if } \max_\kappa \hat p(\kappa \mid f) < \tau_4 .
$$

$\beta$ (`open_set.known_reject_budget`, default 0.10) is the false-alarm budget, an operational policy that is set by the operators and logged with every verdict. A report with very few events, fewer than 24 tokens, is additionally marked as a **short report (possible evasion)**. Sandbox-aware malware that does nothing then remains distinguishable from a genuinely new category.

## 8. Training protocol

```
Input : labelled corpus R (consensus category, per-engine family strings), K folds, beta
Rows  : stratified sample of max_samples reports (default 30,000), stratified by category

for fold k = 1..K (stratified by category):
    for each slot s:
        category slot : fit on the training rows with target = consensus category
        engine slot e : fit on the training rows that engine e labels with a mapped family
                        of support >= min_family_support
        predict the test rows; fold family probabilities to categories (Section 5.3)
    store the out-of-fold category vectors q_s for the test rows

Z <- [q_cat || q_Baidu || q_Kaspersky || q_Symantec]  (out-of-fold, all rows)
evaluate the meta-learner with 5-fold cross_val_predict on Z  -> reported stacking metrics
fit the meta-learner on all of Z                              -> serving meta-learner
tau_4 <- Q_beta(out-of-fold meta confidences)                 -> Eq. 4.11
refit every stand-in slot on all rows                         -> serving slots
```

With RoBERTa slots, fine-tuning has already happened offline. The protocol then scores a subsample of each test fold (`train.roberta_eval_rows`) and fits the stacker on those rows.

The protocol reports the following:

- **labelling strategies** (the structure of thesis Table 4.9): each single-engine model folded to categories and scored against the consensus, the direct category model, soft voting over the four slots, and the stacking meta-learner;
- **per-category precision, recall and F1** of the meta-learner (Table 4.11) and the support-weighted stacking metrics (Table 4.10);
- **family-level metrics** of every engine model on its own labels (Tables 4.4–4.6);
- a six-family view that groups the nine categories as in Table 4.10.

## 9. Category hold-out protocol

This protocol measures generalisation to categories never seen in training (thesis Table 4.7), and it is also the **release gate** of a retrained model (thesis Section 4.4.4). For each category $\kappa^\dagger$:

1. Remove every report of $\kappa^\dagger$ from training. Engine family models also drop the families that map to $\kappa^\dagger$.
2. Train the base slots on 80 % of the remaining reports, and the meta-learner on the other 20 %. Calibrate $\tau_4$ on the 20 % with three-fold out-of-fold predictions.
3. Score the held-out reports and report the share **flagged unknown**, the **nearest known category** and the mean confidence. Also report the known false-unknown rate and the known macro-F1.

The study uses the stand-in heads on the shared feature space. That space is unsupervised, so it contains no label information about $\kappa^\dagger$. By default it runs on a stratified sample of 12,000 reports (`train.holdout_rows`).

## 10. Computational cost of the transformer

For a transformer with $L$ layers, width $d$, $h$ heads and sequence length $n$ (thesis Table 4.14):

$$
\text{MAC}_{\text{forward}} = L\,\big(12\, n\, d^2 + 2\, n^2 d\big), \qquad
\text{attention maps} = L \cdot h \cdot n^2 \cdot 4\ \text{bytes}, \qquad
\text{training} \approx 3 \times \text{forward}.
$$

| Model | $n$ | Parameters | FP32 weights | Forward GMAC | Attention maps per sample |
|---|---|---|---|---|---|
| DistilRoBERTa-base | 512 | 82.1 M | 328 MB | 24.2 | 75.5 MB |
| RoBERTa-base | 128 | 124.6 M | 499 MB | 11.2 | 9.4 MB |
| RoBERTa-base | 256 | 124.6 M | 499 MB | 23.0 | 37.7 MB |
| **RoBERTa-base (thesis)** | **512** | **124.6 M** | **499 MB** | **48.3** | **151 MB** |
| RoBERTa-large | 512 | 355.4 M | 1,421 MB | 167.5 | 403 MB |

The sequence length is the main deployment lever. Halving $n$ roughly halves the compute and quarters the attention memory. The System panel shows these closed forms next to the *measured* size of whatever is loaded in each slot.

## 11. Design rationale

| Decision | Alternatives considered | Reason |
|---|---|---|
| Behaviour reports as input | Static PE features; byte images (Chapter 3); raw API traces | Behaviour survives packing and obfuscation, which defeat static views, so the two chapters cover complementary failure modes. The report is structured, so processing is deterministic and auditable. |
| Consensus of three engines through a category map | One vendor's labels; majority of all engines | Engines disagree strongly (only 50.6 % of multi-engine files agree after mapping). A single vendor's labels teach its blind spots. A small, versioned map is the one component that must be maintained when vendors rename families. |
| Fixed schema, padding and fixed serialisation order | Hand-picked event counts; graph encodings | A stable vocabulary across files; missing information becomes explicit; truncation at $n$ is deterministic; the schema can be versioned. |
| RoBERTa as the encoder | TF-IDF with SVM or tree models; 1-D CNN; LSTM; BERT | Self-attention relates distant events (an early hook, a late network request). Byte-level BPE handles paths, registry keys and API names without an out-of-vocabulary table. Tokens are processed in parallel. |
| `roberta-base`, $n = 512$, batch 8 | `roberta-large`; DistilRoBERTa; other $n$ | 48.3 GMAC per sequence and about 8.3 GB of training memory at batch 8, which fits every GPU used in the thesis, including the 16 GB T4. |
| Stacking over per-engine models and a direct model | Majority voting; the single best model | The meta-learner learns per-category reliability and needs no oracle knowledge of the best engine. |
| τ₄ from a false-alarm budget | A threshold tuned on held-out categories | Uses known data only, and gives a predictable analyst workload. |
| Versioned model, map and schema on every verdict | Unversioned labels | Any decision can be reproduced, and retroactive relabelling can be audited. |

## 12. References

1. Y. Liu et al. *RoBERTa: A robustly optimized BERT pretraining approach.* arXiv:1907.11692, 2019.
2. A. Vaswani et al. *Attention is all you need.* NeurIPS, 2017.
3. D. H. Wolpert. *Stacked generalization.* Neural Networks 5(2): 241–259, 1992.
4. M. Sebastián, R. Rivera, P. Kotzias, J. Caballero. *AVclass: A tool for massive malware labeling.* RAID, 2016.
5. VirusTotal. *API v3 documentation: file behaviour summaries.* https://docs.virustotal.com
