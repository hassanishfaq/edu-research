# Datasets

This folder is intentionally empty in the repository. Place the data here, or point `config.yaml` elsewhere:

```
datasets/
├── Short_Dataset/     28 family folders of 225 x 225 grayscale PNG images   (data.training_images_dir)
└── External/          optional: families never seen by any node            (data.zero_day_dir)
```

The image dataset of the paper is available on FigShare: [doi:10.6084/m9.figshare.29107697](https://doi.org/10.6084/m9.figshare.29107697).

See [docs/DATASETS.md](../docs/DATASETS.md) for details, then run `python scripts/prepare_data.py`.
