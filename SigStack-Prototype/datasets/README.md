# Datasets

This folder is intentionally empty in the repository. Place the behaviour-report archive here, or point `config.yaml` elsewhere:

```
datasets/
└── multi_av_behaviour_reports.zip     <root>/<engine>/<vendor family>/<md5>.json   (data.archive)
```

The report corpus of the thesis was obtained from VirusTotal and is not redistributed. [docs/DATASETS.md](../docs/DATASETS.md) describes the format and how to build a compatible corpus. Then run `python scripts/prepare_data.py`.
