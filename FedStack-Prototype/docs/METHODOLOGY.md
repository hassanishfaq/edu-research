# Methodology

This document gives the formal description of the framework implemented by the prototype. It explains how each equation maps to code, and the rationale behind the main design decisions. Equation numbers in parentheses refer to Chapter 3 of the thesis.

## Contents

1. [Problem formulation](#1-problem-formulation)
2. [Byte-level image representation](#2-byte-level-image-representation)
3. [Class-specialist federation](#3-class-specialist-federation)
4. [Node model](#4-node-model)
5. [Server-side stacking with accuracy-aware weights](#5-server-side-stacking-with-accuracy-aware-weights)
6. [Training protocol](#6-training-protocol)
7. [Open-set decision for unseen families](#7-open-set-decision-for-unseen-families)
8. [Communication model](#8-communication-model)
9. [Computational complexity](#9-computational-complexity)
10. [Design rationale](#10-design-rationale)
11. [The CPU stand-in nodes](#11-the-cpu-stand-in-nodes)
12. [References](#12-references)

---

## 1. Problem formulation

Let $\mathcal{C} = \{c_1, \dots, c_C\}$ be the set of known malware families ($C = 28$ in the thesis). The families are distributed over $K$ nodes, each owned by a different organisation. Node $k$ holds a private labelled set

$$
D_k = \{(x_i, y_i)\}_{i=1}^{|D_k|}, \qquad y_i \in \mathcal{C}_k \subset \mathcal{C},
$$

where $\mathcal{C}_k$ is the node's local label set. Raw samples $x_i$ may not leave node $k$.

The task has two parts:

- **closed-set classification**: learn $f: x \mapsto \hat{y} \in \mathcal{C}$ from the union of the nodes' knowledge without pooling the data;
- **open-set rejection**: decide whether a new sample $x$ belongs to $\mathcal{C}$ at all, and report it as an *unknown family* (a possible zero-day sample) when it does not.

The label sets $\mathcal{C}_k$ are **non-IID by construction**. In the class-specialist setting each node sees only a subset of the families (Section 3). Parameter averaging handles this case poorly, because every node optimises a different output layer.

## 2. Byte-level image representation

A file is never executed. Its bytes $b_1, \dots, b_n \in \{0, \dots, 255\}$ are laid out row by row into a grayscale image whose width $W(n)$ depends on the file size. The width follows the rule of Nataraj et al. [1]:

| File size $n$ | < 10 KB | < 30 KB | < 60 KB | < 100 KB | < 200 KB | < 500 KB | < 1000 KB | ≥ 1000 KB |
|---|---|---|---|---|---|---|---|---|
| Width $W(n)$ | 32 | 64 | 128 | 256 | 384 | 512 | 768 | 1024 |

The height is $H = \lceil n / W \rceil$, and the last row is zero-padded. The image is resized bilinearly to $225 \times 225$, the resolution of the thesis dataset, for display. It is resized to $224 \times 224$ for the backbone. Pixel values are scaled to $[0, 1]$ on a single channel, which is the same transform as torchvision's `ToTensor` on a grayscale image and matches the research code (`fedstack/imaging.py`).

The representation needs no disassembly, runs in linear time and is robust to many code-level obfuscations. Its blind spot is equally clear: files that share a packer produce similar images. The open-set analysis in [EVALUATION.md](EVALUATION.md) shows this effect directly.

The same module computes the Shannon entropy of the byte distribution,

$$
H(X) = -\sum_{v=0}^{255} p_v \log_2 p_v ,
$$

and parses the PE header (machine, subsystem, number of sections, entry point, time stamp and per-section entropy). These values are shown to the analyst. They are **not** used as model inputs.

## 3. Class-specialist federation

The thesis uses $K = 4$ nodes with seven families each (`federation.partition` in `config.yaml`):

| Node | Specialisation | Families |
|---|---|---|
| node_1 | Adware | Imali, InstallRex, InstMonster, MultiPlug (Gen73), Mplug LO4, BrowseFox, Gen Variant Mplug |
| node_2 | Adware / PUA / worms | Graftor, Adware Kazy, Gen Variant MultiPlug, Application Bundler, Application Kazy, Mydoom (email worm), Allaple (net worm) |
| node_3 | PUA / Trojans | PUA Multiplug, PUA Outbrowse, PUA Somoto, Trojan Agent BDMJ, TR Agent, TR Crypt XPACK, TR Dropper |
| node_4 | Win32 file infectors | Cryptolocker C, Parite B, Parite C, Ramnit, Sality 3, Virlock, Virtob |

This partition models a realistic federation, in which each organisation specialises in the threats it actually encounters. It is also the hardest case for parameter averaging: the nodes do not even share an output space.

## 4. Node model

Each node owns an `EmbeddingNet` (`fedstack/models.py`). The class is identical to the one in the research code, so checkpoints written by `train_node.py` load unchanged. It has two parts:

- a **backbone** $g_k$: ResNet-50 in the thesis configuration (ResNet-18 and AlexNet are also supported), with a single-channel first convolution, followed by global average pooling that yields a 2048-dimensional embedding (Eq. 3.9);
- a **linear head** $h_k$ with $|\mathcal{C}_k| = 7$ outputs and a soft-max (Eq. 3.10).

The node output for an image $x$ is

$$
p_k(x) = \operatorname{softmax}\big(h_k(g_k(x))\big) \in \Delta^{|\mathcal{C}_k|-1}.
$$

Node training minimises the empirical cross-entropy on the node's own data (Eq. 3.12):

$$
\theta_k^{\ast} = \arg\min_{\theta} \frac{1}{|D_k|} \sum_{(x,y) \in D_k} \mathcal{L}_{\mathrm{CE}}\big(p_k(x;\theta), y\big),
$$

using Adam (learning rate $10^{-3}$, weight decay $10^{-5}$), batch size 64 and $E = 5$ local epochs per round, over $T = 20$ rounds (thesis Section 3.5.1). The backbone is initialised from ImageNet weights (transfer learning).

## 5. Server-side stacking with accuracy-aware weights

### 5.1 Class alignment

Node $k$ returns a vector over its local classes. The server embeds it into the global class space with an injection $\pi_k: \mathcal{C}_k \to \mathcal{C}$:

$$
\tilde{p}_k(x)[c] = \begin{cases} p_k(x)[\pi_k^{-1}(c)] & c \in \pi_k(\mathcal{C}_k) \\ 0 & \text{otherwise.} \end{cases}
$$

The global vocabulary is the union of the node vocabularies, built exactly as in `server_stack.py` (`Engine.align`).

### 5.2 Accuracy-aware node weights

Let $a_k^{(t)}$ be the accuracy of node $k$ on its local validation split after round $t$. The weights for the next round are (Eq. 3.15):

$$
w_k^{(t+1)} = \frac{a_k^{(t)}}{\sum_{j=1}^{K} a_j^{(t)}}, \qquad \sum_k w_k = 1 .
$$

Nodes that learned their families well therefore contribute more to the stacked representation, and weak nodes contribute less (`Engine._weights`). A uniform weighting is available through `model.accuracy_weighting: false`.

### 5.3 Stacked representation and meta-learner

The meta-learner input is the concatenation of the weighted, aligned node outputs:

$$
z(x) = \big[\, w_1 \tilde{p}_1(x) \;\|\; w_2 \tilde{p}_2(x) \;\|\; \cdots \;\|\; w_K \tilde{p}_K(x) \,\big] \in \mathbb{R}^{K C} .
$$

With $K = 4$ and $C = 28$, $z$ has 112 dimensions. The meta-learner $S$ is fitted on a validation pool $V$ held by the server, whose images were never used to train any node:

$$
S^{\ast} = \arg\min_{S} \sum_{(x,y) \in V} \mathcal{L}_{\mathrm{CE}}\big(S(z(x)), y\big), \qquad \hat{y}(x) = \arg\max_{c} S(z(x))[c] .
$$

The thesis uses multinomial logistic regression, with $(KC + 1)\,C = 3{,}164$ parameters. The prototype also offers a random forest (`model.stacker: rf | logreg`), which is its default because it is more robust with the weaker stand-in nodes. The meta-learner learns *where* each node is reliable, including which families a node confuses with which, so it can correct the systematic errors of individual nodes rather than averaging them away.

## 6. Training protocol

```
Input : node datasets D_1..D_K, backbone, meta-learner S, rounds T, local epochs E
Output: node models M_1..M_K, weights w, meta-learner S, open-set threshold tau

for t = 1..T:
    for each node k (in parallel at the participants):
        train M_k on D_k for E epochs                         # raw data stays local
        a_k  <- accuracy of M_k on its local validation split
        send INT8(theta_k) and the prediction matrix P_k      # Section 8
    w_k <- a_k / sum_j a_j                                    # Eq. 3.15
    Z   <- [ w_1 align(M_1(V)) || ... || w_K align(M_K(V)) ]  # server validation pool
    fit S on (Z, y_V); report test accuracy and macro-F1
tau <- calibrate(S, Z, y_V, beta)                             # Section 7 (after the last round)
```

In the prototype, `Engine._federation_surrogate` runs this loop live with the stand-in nodes. With trained ResNet checkpoints, node fine-tuning has already happened offline on a GPU, so `Engine._federation_resnet` replays the server side: node inference on the validation pool, the weights, the meta-learner fit and the payload accounting.

## 7. Open-set decision for unseen families

A closed-set classifier always outputs one of the known families. The prototype makes it an open-set detector with a threshold on the meta-learner confidence (thesis Eq. 3.17):

$$
s(x) = \max_{c} \hat{p}(c \mid x), \qquad \tau = Q_{\beta}\big\{\, s(x) : x \in V_{\text{known}} \,\big\},
$$

where $Q_\beta$ is the $\beta$-quantile and $V_{\text{known}}$ is the server validation pool. A sample is reported as **unknown family (possible zero-day)** when $s(x) < \tau$, together with its nearest known family for triage.

Two details matter:

- **Out-of-fold calibration.** The confidences $\hat{p}$ used for $\tau$ are obtained by five-fold `cross_val_predict` on $V_{\text{known}}$. Confidences of a meta-learner scored on its own training data would be inflated and would place $\tau$ too high.
- **β is policy, not a tuned parameter.** $\beta$ is the share of known samples that the operator accepts to see flagged, which is the false-alarm budget. Unseen families are never used to choose $\tau$, because they are not available before deployment. The prototype still reports a threshold sweep on them, to show the trade-off.

The normalised-entropy score $1 - H(\hat{p}) / \log C$ is available as an alternative (`open_set.method: entropy`).

## 8. Communication model

In round $t$ each node uploads two objects:

1. its weights $\theta_k$, quantised to 8 bits with one FP32 scale per tensor, $q = \operatorname{round}\big(\theta / s\big)$ with $s = \max|\theta| / 127$;
2. its soft-max prediction matrix on its local validation split, $P_k \in \mathbb{R}^{n_k^{\text{val}} \times C}$ in FP32.

Nothing is downloaded, because there is no global model to broadcast; the meta-learner lives only at the server. The upload per node and round is therefore

$$
B_k^{\text{FedStack}} = |\theta_k| \cdot 1\,\text{B} + n_k^{\text{val}} \cdot C \cdot 4\,\text{B},
\qquad
B_k^{\text{FedAvg}} = 2 \cdot |\theta_k| \cdot 4\,\text{B}\ \ (\text{FP32 up and down}).
$$

For ResNet-50 the ratio is close to $8\times$ in favour of FedStack. The prototype measures these quantities on **serialised** state dictionaries (`models.state_dict_bytes`), not analytically, and reports them per node, per round and for all $T$ rounds (`Engine.payload_round`).

## 9. Computational complexity

| Stage | Cost | Where it runs |
|---|---|---|
| Byte imaging | $O(n)$ in the file size, plus a constant-size resize | endpoint |
| Node inference | $K$ backbone forwards, 4.1 GMAC each for ResNet-50 at $224^2$ | nodes (CPU or GPU) |
| Alignment and weighting | $O(KC)$ | server |
| Meta-learner, logistic regression | $O(KC \cdot C)$ multiply-accumulates (about 3,100) | server |
| Meta-learner, random forest | $O(N_{\text{trees}} \cdot \text{depth})$ | server |
| Round, node side | $E \cdot \lvert D_k \rvert$ forward and backward passes | participants (GPU) |
| Round, server side | $K \cdot \lvert V \rvert$ forward passes plus the meta-learner fit | server |
| Open-set calibration | five meta-learner fits on $V$, once per federation | server |

Adding a node grows the meta-learner input linearly ($KC$). The per-node training cost does not change, because each node trains only on its own data.

## 10. Design rationale

| Decision | Alternatives considered | Reason |
|---|---|---|
| Byte images as input | Hand-crafted static features; disassembly; dynamic traces | Execution-free, linear-time and robust to many code obfuscations; one pipeline for any file type. Behaviour-based detection is the complementary view taken in Chapter 4. |
| Class-specialist nodes | Random (IID) partition | Models real organisations, which see different threats; it is the regime in which parameter averaging degrades most. |
| Stacking instead of parameter averaging | FedAvg, FedProx; majority voting | Nodes with disjoint label sets cannot be averaged meaningfully. Stacking learns how reliable each node is per family, and uploads are dominated by quantised weights sent once per round rather than twice. |
| Accuracy-aware weights (Eq. 3.15) | Uniform weights; data-size weights | Down-weights nodes whose local model is weak, independently of how much data they hold. |
| Logistic-regression meta-learner | Random forest; MLP | Only 3,164 parameters, a 0.3 ms decision and calibrated probabilities. The random forest is offered for weaker base models. |
| INT8 weights per round | FP32 or FP16; predictions only | A 4× smaller upload than FP32 without affecting the server's use of the weights. Predictions-only would be cheaper still but would lose the ability to re-run a node at the server. |
| Threshold from a false-alarm budget | Threshold tuned on unseen families; OpenMax; energy scores | Uses known data only and needs no retraining. The analyst workload it creates is predictable. |
| Configuration-driven model slot | Hard-coded backbone | The same software runs with stand-ins on a laptop and with trained checkpoints in production. |

## 11. The CPU stand-in nodes

Until trained ResNet checkpoints are available, each node is a `SurrogateNode` (`fedstack/surrogate.py`) with the same "frozen feature extractor + trainable head" structure as the research code with `freeze_backbone: true`:

- the **feature extractor** is standardisation followed by PCA (96 components by default), fitted only on the node's own 64 × 64 thumbnails, which plays the role of the frozen backbone;
- the **head** is a one-hidden-layer neural network (256 units, Adam) trained incrementally with `partial_fit`, one call per local epoch, so every federated round is a real local training step with a visible learning curve.

The stand-ins exercise the complete system: rounds, weighting, stacking, payload accounting, open-set calibration, scanning and the watcher. They are labelled as stand-ins throughout the dashboard. Their accuracy is **not** an estimate of the thesis model.

## 12. References

1. L. Nataraj, S. Karthikeyan, G. Jacob, B. S. Manjunath. *Malware images: visualization and automatic classification.* Proceedings of the 8th International Symposium on Visualization for Cyber Security (VizSec), 2011.
2. H. Ishfaq, J. H. Shah, R. Saleem, M. Afzal. *A distributed framework for zero-day malware detection using federated ensemble models.* PLOS ONE 21(1): e0339907, 2026. doi:10.1371/journal.pone.0339907
3. H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. *Communication-efficient learning of deep networks from decentralized data.* AISTATS, 2017 (FedAvg).
4. D. H. Wolpert. *Stacked generalization.* Neural Networks 5(2): 241–259, 1992.
5. K. He, X. Zhang, S. Ren, J. Sun. *Deep residual learning for image recognition.* CVPR, 2016.
