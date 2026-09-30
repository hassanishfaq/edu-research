# Datasets

CAQ-EdgeNet combines two modalities (thesis Table 5.1): **network flows** (CIC-IoT-2023) and **device context** (MDF-style context data with the busy-state target). No data are included in this repository. Without them the prototype runs on a clearly labelled **synthetic stand-in** and a **device-context simulator**.

| Dataset (thesis) | Samples | Features | Target |
|---|---|---|---|
| CIC-IoT-2023 | 62,451 (sub-sampled to the counts of Table 5.2) | 46 flow features | 8 classes |
| MDF context data | 48,320 | 7 context features | busy state |
| Combined | 110,771 | 53 | multi-task |

## 1. CIC-IoT-2023

CIC-IoT-2023 [1] was published by the Canadian Institute for Cybersecurity. It contains 33 attacks executed by malicious IoT devices against 105 IoT devices, released as per-window flow features in CSV files. It is available from the institute's dataset page (https://www.unb.ca/cic/datasets/iotdataset-2023.html) under its terms of use.

### Expected layout

```
datasets/
└── CIC-IoT-2023/
    ├── part-00000-...csv
    ├── part-00001-...csv
    └── ...
```

Every CSV must contain the 46 feature columns listed in `caqedge/data.py: FLOW_FEATURES` and a `label` column. These are the column names of the released CSV files (`flow_duration`, `Header_Length`, `Protocol Type`, …, `Weight`). Up to `data.max_rows_per_file` rows are read per file.

### Label mapping to the eight classes of thesis Table 5.2

| Class | CIC-IoT-2023 labels |
|---|---|
| Benign | `BenignTraffic` |
| DDoS | `DDoS-RSTFINFlood`, `DDoS-PSHACK_Flood`, `DDoS-SYN_Flood`, `DDoS-UDP_Flood`, `DDoS-TCP_Flood`, `DDoS-ICMP_Flood`, `DDoS-SynonymousIP_Flood`, `DDoS-ACK_Fragmentation`, `DDoS-UDP_Fragmentation`, `DDoS-ICMP_Fragmentation`, `DDoS-SlowLoris`, `DDoS-HTTP_Flood` |
| DoS | `DoS-UDP_Flood`, `DoS-SYN_Flood`, `DoS-TCP_Flood`, `DoS-HTTP_Flood` |
| Reconnaissance | `Recon-PingSweep`, `Recon-OSScan`, `Recon-PortScan`, `Recon-HostDiscovery`, `VulnerabilityScan` |
| Mirai | `Mirai-greeth_flood`, `Mirai-greip_flood`, `Mirai-udpplain` |
| Spoofing | `MITM-ArpSpoofing`, `DNS_Spoofing` |
| Brute Force | `DictionaryBruteForce` |
| Web Attack | `BrowserHijacking`, `Backdoor_Malware`, `CommandInjection`, `SqlInjection`, `XSS`, `Uploading_Attack` |

After mapping, each class is sub-sampled at random (seed 42) to the count in `data.class_counts`, which defaults to thesis Table 5.2: 18,240 · 12,531 · 8,312 · 6,821 · 5,213 · 4,915 · 3,928 · 2,491. This gives an imbalance ratio of 7.32 : 1.

**Pre-processing.** $\log(1 + \max(x, 0))$ followed by standardisation with the mean and standard deviation of the training split. The split is stratified 70 / 15 / 15.

## 2. Synthetic stand-in flows

When `data.cic_iot_dir` holds no CSV files, `synth_flows` generates flows with the **same 46 feature names and class counts**:

- **Scale features** (rates, header length, duration, packet statistics, inter-arrival time) are log-normal around class-specific centres. DDoS and Mirai have very high rates and short flows; spoofing has long, slow flows; brute force and web attacks have larger payloads. A per-sample intensity factor correlates rate-type features within a flow.
- **Flag and protocol indicators** are window fractions. They are drawn as Bernoulli values with class-specific rates (SYN floods, UDP-only Mirai, ARP spoofing, SSH/Telnet brute force, HTTP web attacks), with some intermediate fractions.
- **Label noise** of 1.2 % adds realistic overlap.

The stand-in exercises the system. It does **not** reproduce the statistics of CIC-IoT-2023, and results on it are not comparable with the thesis.

## 3. Device context and the busy-state target

`DeviceContextSim` produces, per device and interval, the seven context features: battery, CPU, memory, network activity, active applications, and time of day as $(\sin, \cos)$. It has the following properties:

- **Regime switching.** Quiet periods alternate with busy bursts of 3–12 intervals. A burst is announced one interval in advance by the leading indicators an application launch produces: rising network activity and one more active application.
- **Battery.** The battery drains in proportion to CPU load and to the energy of the requests served, and charging episodes occur when the battery is low.
- **Target.** The busy-state target is the CPU load of the next interval.

A CSV with the columns `battery, cpu, memory, net_activity, active_apps, hour_sin, hour_cos` and optionally `busy_next` can replace the simulator (`data.context_csv`). Rows are sampled to match the number of flows.

## 4. Cache

The assembled dataset is cached in `data/cache/dataset.npz`, about 16 MB. It holds the raw and standardised flows, the context, labels, busy targets, the split, the standardisation statistics and the data source. Delete the file to rebuild it, for example after adding the CIC-IoT-2023 CSVs.

## 5. Ethics and licensing

- CIC-IoT-2023 is subject to the terms of the Canadian Institute for Cybersecurity. Cite [1] when you use it.
- The prototype captures no network traffic and executes no code. Handle raw packet captures under your institution's data-protection rules.

## References

1. E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani. *CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment.* Sensors 23(13): 5941, 2023. doi:10.3390/s23135941
