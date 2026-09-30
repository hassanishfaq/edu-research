# Methodology

This document gives the formal description of CAQ-EdgeNet as implemented by the prototype, the cost model and the Markov decision process of the offloading controller, and the rationale behind the main design decisions. Equation numbers in parentheses refer to Chapter 5 of the thesis.

## Contents

1. [Problem formulation](#1-problem-formulation)
2. [Feature aggregation](#2-feature-aggregation)
3. [Multi-modal encoder and fusion](#3-multi-modal-encoder-and-fusion)
4. [Conditional attention](#4-conditional-attention)
5. [Quantisation-aware compression](#5-quantisation-aware-compression)
6. [Heads and multi-task training](#6-heads-and-multi-task-training)
7. [Hybrid, sensitivity-aware deployment precision](#7-hybrid-sensitivity-aware-deployment-precision)
8. [Execution paths and cost model](#8-execution-paths-and-cost-model)
9. [Offloading as a Markov decision process](#9-offloading-as-a-markov-decision-process)
10. [Controllers and baselines](#10-controllers-and-baselines)
11. [Operating the controller safely](#11-operating-the-controller-safely)
12. [Design rationale](#12-design-rationale)
13. [References](#13-references)

---

## 1. Problem formulation

A fleet of IIoT and mobile devices must classify its own network traffic into $K = 8$ classes. These are benign traffic and the seven attack classes of CIC-IoT-2023 grouped as in thesis Table 5.2: DDoS, DoS, Reconnaissance, Mirai, Spoofing, Brute Force and Web Attack. Each request, which is one flow window, can be served in one of three places:

- **local**: the whole model runs on the device;
- **partial**: the device runs the encoder and sends a compact quantised embedding, and the edge server runs the classifier;
- **edge**: the device sends the raw features, and the edge server runs the whole model.

The objective combines three things: detection quality, latency and energy. The best placement depends on a device state that changes every interval. CAQ-EdgeNet therefore learns a detector that is cheap enough to run locally, together with a controller that chooses the placement per request.

## 2. Feature aggregation

Each request carries two modalities (Eq. 5.1), $X = \{x_d, x_s\}$:

- $x_d \in \mathbb{R}^{46}$: the **dynamic, network-level** features of a CIC-IoT-2023 flow window. They include flow duration, header length, protocol type, rates, TCP flag counts, application-protocol indicators (HTTP, HTTPS, DNS, Telnet, SSH, …) and packet-size statistics. They are transformed with $\log(1 + \max(x, 0))$ and standardised with training-split statistics.
- $x_s \in \mathbb{R}^{7}$: the **static / device-context** features of the MDF-style context data. These are battery level, CPU load, memory, network activity, the number of active applications, and the time of day as $(\sin, \cos)$.

The **busy-state target** $y_b \in [0, 1]$ is the device CPU load in the **next** interval. The device state that conditions the model and the controller is (Eq. 5.5)

$$
S_t = \big(\text{battery}_t,\ \text{CPU}_t,\ \text{latency}_t\big) \in [0,1]^3 ,
$$

where latency is the observed round-trip time to the edge service (link RTT plus edge service time) divided by 250 ms and clipped to 1.

## 3. Multi-modal encoder and fusion

Each modality is embedded separately, because the two modalities differ in scale and statistics (Eq. 5.2):

$$
h_d = \sigma(W_d x_d + b_d) \in \mathbb{R}^{64}, \qquad h_s = \sigma(W_s x_s + b_s) \in \mathbb{R}^{16},
$$

and then fused (Eq. 5.3):

$$
z = \sigma\big(W_f\,[h_d ; h_s] + b_f\big) \in \mathbb{R}^{64},
$$

with $\sigma = \mathrm{ReLU}$. The fused embedding $z$ is the natural **cut point** of the partial path. It is compact, and it does not expose the raw features.

## 4. Conditional attention

A feature gate conditioned on the device state modulates the embedding (the *FusionCAB* block):

$$
a = \mathrm{sigmoid}\big(W_a\,[z ; S_t] + b_a\big) \in (0,1)^{64}, \qquad z' = 2\,a \odot z .
$$

The factor 2 makes $a = 0.5$ the identity, so the gate can amplify or attenuate each feature. The same flow can therefore be processed differently under different resource budgets. During training, $S_t$ is drawn per sample: the battery and CPU values come from the sample's context row, and the latency from $\min(\mathrm{Exp}(0.25), 1)$. The gate thus sees the full range of operating conditions.

## 5. Quantisation-aware compression

The attended embedding is quantised to $b$ bits (Eq. 5.4):

$$
s = \frac{\max z' - \min z'}{2^b - 1}, \qquad \hat z = \operatorname{round}\!\left(\frac{z'}{s}\right) s .
$$

During training the rounding is simulated with a **straight-through estimator**: the forward pass uses $\hat z$, and the backward pass treats $\partial \hat z / \partial z' = 1$. The classifier therefore learns to be accurate on the quantised embedding that the partial path actually transmits. With $b = 8$ and $d_z = 64$, the partial path sends $64 + 4 = 68$ bytes (the codes plus one FP32 scale). The edge path sends $53 \times 4 = 212$ bytes of raw features.

## 6. Heads and multi-task training

- **Classifier**: $\hat z \rightarrow 128 \rightarrow 64 \rightarrow 8$ with ReLU activations and a soft-max.
- **Busy-state head**: $z \rightarrow 16 \rightarrow 1$ with a sigmoid output, which forecasts $y_b$.

Both heads share the encoders (multi-task shared learning). The training loss is

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}\big(\hat y, y\big) + \lambda_{\text{busy}}\ \mathrm{MSE}\big(\hat y_b, y_b\big), \qquad \lambda_{\text{busy}} = 1.0,
$$

and it is minimised with Adam (learning rate $2\times10^{-3}$, weight decay $10^{-4}$) with cosine annealing, batch size 256 and 12 epochs, on a stratified 70 / 15 / 15 split. The model has 30,825 parameters.

## 7. Hybrid, sensitivity-aware deployment precision

The trained weights are executed by a numpy runtime (`NumpyRuntime`) at a per-layer precision $b_\ell \in \{32, 16, 8, 4\}$:

- **32 bits**: FP32 weights and activations.
- **16 bits**: FP16 weights.
- **8 or 4 bits**: symmetric per-tensor weight codes, $w_q = \operatorname{clip}\big(\operatorname{round}(w / s_w), \pm(2^{b_\ell - 1} - 1)\big)$ with $s_w = \max|w| / (2^{b_\ell - 1} - 1)$. Activations are quantised to 8 bits, products are accumulated in int32, and the result is rescaled by $s_x s_w$.

**Sensitivity analysis.** For every layer $\ell$ the accuracy drop is measured with only that layer quantised:

$$
\Delta_\ell(b) = \mathrm{Acc}_{\text{FP32}} - \mathrm{Acc}\big(\ell \text{ at } b \text{ bits}\big).
$$

**Hybrid rule** (Section 5.2.3 of the thesis):

1. The attention gate and the final classifier layer are kept at 16 bits, because they are the critical layers.
2. Any layer with $\Delta_\ell(8) > 0.3$ points is kept at 16 bits.
3. The busy-head layers with $\Delta_\ell(4) \le 0.1$ points drop to 4 bits.
4. All other layers use 8 bits.

The size of an export is $\sum_\ell \big(\lceil b_\ell\,|W_\ell| / 8 \rceil + 4\,|b_\ell^{\text{bias}}| + 4\big)$ bytes.

**Device latency.** Numpy has no integer kernels, so the numpy runtime reproduces the sizes and accuracies of each precision but not the speed-ups of a mobile integer runtime. The device inference time is therefore anchored:

$$
t_{\text{dev}}(p) = \frac{t^{\text{host}}_{\text{FP32}} \cdot \kappa}{\mathrm{speedup}(p)}, \qquad \kappa = \frac{12.6\ \text{ms}}{t^{\text{host}}_{\text{FP32}}},
$$

where 12.6 ms is the on-device FP32 time of thesis Table 5.7. The speed-ups (1.0, 1.35, 2.33, 2.2 for FP32, FP16, INT8 and hybrid) are the ratios of that table. An explicit `device_profile.compute_factor` replaces $\kappa$.

## 8. Execution paths and cost model

Let $\text{slow} = 1 + \kappa_{\text{cpu}}\,\text{CPU}^2$ ($\kappa_{\text{cpu}} = 25$) be the stretching of on-device work by the device's own load. Let $t_{\text{tx}}(B) = 8B/\text{bw} + 2$ ms be the transmission time of $B$ bytes, and $t_{\text{edge}}(\text{load}) = t^{0}_{\text{edge}}\,(1 + 3\,\text{load})$ the service time at the shared edge server. The service time at zero load, $t^0_{\text{edge}}$, is 28 ms for the full model and 3 ms for the heads.

| Path | Latency $L$ | Energy $E$ | Bytes sent |
|---|---|---|---|
| local | $t_c + t_{\text{dev}}^{\text{full}}\cdot\text{slow}$ | $P_a t_c + P_i t^{\text{full}}_{\text{dev}} + P_0\,(\text{slow}-1)\,t^{\text{full}}_{\text{dev}}$ | 0 |
| partial | $t_c + t_{\text{dev}}^{\text{enc}}\cdot\text{slow} + t_s + t_{\text{tx}}(68) + \text{RTT} + t^{\text{heads}}_{\text{edge}}$ | $P_a(t_c + t_s) + P_i t^{\text{enc}}_{\text{dev}} + P_r t_{\text{tx}} + P_0\,(\ldots)$ | 68 |
| edge | $t_c + t_s + t_{\text{tx}}(212) + \text{RTT} + t^{\text{full}}_{\text{edge}}$ | $P_a(t_c + t_s) + P_r t_{\text{tx}} + P_0(\text{RTT} + t_{\text{edge}})$ | 212 |

Here $t_c$ is the collection time, $t_s$ the serialisation time, and $P_a, P_i, P_r, P_0$ the active, inference, radio and idle power. All are parameters of `config.yaml` (`device_profile`, `network`). Each path is also evaluated for detection correctness at the precision it executes: INT8 hybrid on the device, INT8 encoder with FP32 heads, or FP32 at the edge.

## 9. Offloading as a Markov decision process

- **State.** $s_t = (\text{battery}, \text{CPU}, \text{latency}, \text{bandwidth}/10, \text{link up}, \hat y_b, \text{charging}) \in [0,1]^7$, the device state $S_t$ of Eq. 5.5 extended with the link, the busy-state forecast of the detector and the charging flag.
- **Actions.** $a_t \in \{\text{local}, \text{partial}, \text{edge}\}$ (Eq. 5.6), with a **feasibility mask** that removes partial and edge when the link is down.
- **Dynamics.** The device context follows a regime-switching process: quiet periods, then application launches announced one interval ahead by network activity, then busy bursts of 3–12 intervals. The link has outages (probability 0.03, lasting 3–12 steps) and congestion episodes (probability 0.05, lasting 2–8 steps, with RTT 140 ms and a quarter of the bandwidth). The edge server's load follows an autoregressive process. The decision is taken at $t$, but the request is served during the next interval, so the path is billed at the load in force *then*. This is what makes a forecast of the load useful.
- **Reward** (Eq. 5.7):

$$
r_t = \lambda_1\,\mathbb{1}[\text{correct}] - \lambda_2\,\frac{L_t}{L_{\text{ref}}} - \lambda_3\,\frac{E_t}{E_{\text{ref}}}, \qquad (\lambda_1, \lambda_2, \lambda_3) = (1, 0.6, 0.6),\ L_{\text{ref}} = 150\ \text{ms},\ E_{\text{ref}} = 0.15\ \text{J}.
$$

- **Objective** (Eq. 5.8): $J(\pi) = \mathbb{E}_\pi\big[\sum_{t=0}^{T} \gamma^t r_t\big]$ with $\gamma = 0.95$ and episodes of $T = 120$ requests.

## 10. Controllers and baselines

**Proposed: deep Q-network.**

| Setting | Value |
|---|---|
| Q-network | $7 \to 64 \to 64 \to 3$ with ReLU (4,867 parameters) |
| Replay buffer | 30,000 transitions; updates start after 512 transitions; minibatch 64 |
| Loss | Huber |
| Target network | synchronised every 250 steps |
| Masked target | $y = r + \gamma \max_{a' \in \mathcal{A}_{\text{feasible}}(s')} Q_{\text{target}}(s', a')$ |
| Exploration | ε decays linearly from 1.0 to 0.05 |
| Optimiser | Adam, learning rate $10^{-3}$ |
| Training | 200 episodes × 120 steps |

**Baselines** (the structure of thesis Table 5.4):

| Policy | Rule |
|---|---|
| Random | uniform over feasible actions |
| Rule-based | edge if battery < 0.30 and the link is up; partial if CPU > 0.60 and the link is up; otherwise local. It is also the deterministic fallback. |
| Cost threshold | offload to the cheapest path under the *nominal* link only if its expected cost is at most 85 % of the local cost |
| Greedy offloading | the action with the highest immediate expected reward under the nominal link and the measured per-path accuracy; no forecast and no look-ahead |
| Q-learning (tabular) | discretised state (battery, CPU, latency, link, forecast), learning rate 0.15 |
| DQN without busy forecast | same as the proposed controller, with the forecast replaced by the current CPU load |

**Clairvoyant oracle.** At every step the oracle takes the feasible action with the highest *realised* reward. It knows the jitter, congestion and load that no causal policy can observe. **Decision accuracy** is the share of steps on which a policy agrees with the oracle. **Efficiency** is the policy's mean reward divided by the oracle's.

## 11. Operating the controller safely

The live fleet applies the operating rules of thesis Section 5.4.4:

1. **Measure the reward.** Latency and energy are recorded per request, and the reward is computed from the log, never estimated on the device.
2. **Mask infeasible actions** and fall back deterministically. When the link is down only local execution is allowed; if the local runtime fails, the edge path is used.
3. **Hysteresis.** A switch to a new action requires the policy's preference to persist for two consecutive decisions (`controller.hysteresis`).
4. **Shadow mode.** The rule-based policy is evaluated on the same requests without acting, so its reward, latency and energy can be compared with the released policy.
5. **Versioning.** Every telemetry record carries the detector, quantised-export and policy versions.

## 12. Design rationale

| Decision | Alternatives considered | Reason |
|---|---|---|
| Two modalities, encoded separately and then fused | A single modality; early concatenation | IoT attacks are visible in the traffic, and the device's own behaviour decides where to run. Separate encoders respect the different statistics, and the fused $z$ is the natural split point. |
| Attention conditioned on the device state | Unconditional attention; none | The same input deserves different processing under different budgets. |
| Busy-state head trained jointly | React to the current load; a separate forecaster | Offloading must be decided before the device becomes busy. Sharing the encoder costs almost nothing. |
| Hybrid, sensitivity-aware quantisation | Uniform INT8; FP16 only | It protects the layers where precision matters and compresses the rest. |
| Three actions including *partial* | Binary local/edge | The partial path keeps raw features on the device, sends 68 B instead of 212 B, and uses the edge's spare capacity when the device is busy. |
| Learned controller with an explicit reward | Static thresholds; greedy cost minimisation | The accuracy/latency/energy trade-off changes with the state. The reward weights $\lambda$ are the operator's policy knobs. |
| A nearby edge server as the offloading target | Cloud | The round-trip time and radio energy dominate the cost of offloading. |

## 13. References

1. E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani. *CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment.* Sensors 23(13): 5941, 2023.
2. V. Mnih et al. *Human-level control through deep reinforcement learning.* Nature 518: 529–533, 2015.
3. B. Jacob et al. *Quantization and training of neural networks for efficient integer-arithmetic-only inference.* CVPR, 2018.
4. Y. Bengio, N. Léonard, A. Courville. *Estimating or propagating gradients through stochastic neurons for conditional computation.* arXiv:1308.3432, 2013.
5. Y. Kang et al. *Neurosurgeon: Collaborative intelligence between the cloud and mobile edge.* ASPLOS, 2017.
6. Y. Mao, C. You, J. Zhang, K. Huang, K. B. Letaief. *A survey on mobile edge computing: The communication perspective.* IEEE Communications Surveys & Tutorials 19(4): 2322–2358, 2017.
