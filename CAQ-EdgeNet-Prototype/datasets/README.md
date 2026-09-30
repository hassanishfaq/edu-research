# Datasets

This folder is intentionally empty in the repository. To use the real flow data, place the CIC-IoT-2023 CSV files here:

```
datasets/
└── CIC-IoT-2023/        *.csv with the 46 flow features and a "label" column   (data.cic_iot_dir)
```

CIC-IoT-2023 is available from the Canadian Institute for Cybersecurity: https://www.unb.ca/cic/datasets/iotdataset-2023.html

Without it, the prototype generates a synthetic stand-in with the same feature names and class counts. Delete `data/cache/dataset.npz` after adding the CSVs. See [docs/DATASETS.md](../docs/DATASETS.md).
