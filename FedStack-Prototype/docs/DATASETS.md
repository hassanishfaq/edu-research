# Datasets

The prototype needs **images of malware**, not malware binaries. No data are included in this repository. Datasets are distributed separately under their own terms.

## 1. Private dataset of the paper (28 families)

| Property | Value |
|---|---|
| Origin | PE files collected from public repositories and validated by three antivirus engines through VirusTotal (paper Section 3.1) |
| Representation | 8-bit grayscale PNG, 225 × 225 pixels, produced with the byte-imaging rule of [METHODOLOGY.md](METHODOLOGY.md#2-byte-level-image-representation) |
| Classes | 28 families: Adware, PUA, Trojan, worm and Win32 file-infector families |
| Size used by the prototype | 14,000 images (500 per family), the training split of the paper |
| Availability | FigShare, [doi:10.6084/m9.figshare.29107697](https://doi.org/10.6084/m9.figshare.29107697) |

### Expected layout

```
datasets/
└── Short_Dataset/
    ├── Adware_ADWARE_Imali_Gen0/        *.png
    ├── Adware_ADWARE_InstallRex_Gen2/   *.png
    ├── ...                              (28 family folders)
    └── Win32_Virtob_Gen_12/             *.png
```

Folder names are normalised by removing whitespace, so a folder called `Adware_ADWARE_Imali_Gen 0` is accepted. The names must match `federation.partition` in `config.yaml`. If the downloaded archive is organised differently, point `data.training_images_dir` at the folder that contains the 28 family sub-folders.

## 2. Families unseen by every node (optional)

The zero-day panel needs families that no node was trained on. The thesis evaluation used five modern families collected after the training set: **Danabot, Dridex, Gozi, RecordBreaker and RedLine Stealer** (12, 28, 120, 120 and 120 images). Any set of family folders of images works:

```
datasets/
└── External/
    ├── Danabot/        *.png
    ├── Dridex/         *.png
    └── ...
```

`scripts/prepare_data.py` keeps up to 120 images per unseen family, taking the smallest files first to keep the cache compact. When `datasets/External` is absent, the zero-day panel stays empty and everything else works.

## 3. Public benchmark (for comparison)

The paper also reports results on **Malimg** (25 families, 9,339 images) [1]. The prototype reads any folder of family sub-folders of images. To try it, point `data.training_images_dir` at Malimg and adapt `federation.partition` to its family names.

## 4. Preparation pipeline

`python scripts/prepare_data.py` performs the following steps:

1. For every family, draws a deterministic permutation (seed 42) and assigns 350 images to training, 75 to validation and 75 to test (`data.split`).
2. Converts every selected image to a 64 × 64 thumbnail (`data.thumbnail_size`). The stand-in nodes use these thumbnails.
3. Writes `data/cache/thumbs64.npz` containing the thumbnails, labels, split codes, file names, class names and the unseen-family thumbnails. The file is about 34 MB for the full dataset.
4. Copies a few full-resolution test images per family to `data/samples/` for the dashboard's sample strip.

The ResNet nodes do not use the thumbnails. They read the **original images** from `datasets/` at 224 × 224 and keep them in memory as 8-bit arrays after first use, about 50 KB per image. The dataset folders must therefore stay in place when trained checkpoints are used. If they are missing, the engine falls back to up-sampled thumbnails and logs a warning, because that lowers the accuracy of a trained model.

## 5. Scanning your own files

Any file can be dropped on the dashboard: executables, DLLs, PDFs or arbitrary binaries. It is read as bytes and converted with the same imaging rule. Images (PNG, JPEG, BMP) are treated as already converted. Files are never executed.

## 6. Ethics and licensing

- The images are **derived representations**. They cannot be executed and cannot be converted back to a working binary without the original byte order and size.
- Respect the licence of each dataset. The private dataset is released on FigShare under the terms given there; Malimg is distributed by its authors.
- When you scan real samples, do it inside an isolated analysis environment. Do not store live malware in a repository.

## References

1. L. Nataraj, S. Karthikeyan, G. Jacob, B. S. Manjunath. *Malware images: visualization and automatic classification.* VizSec, 2011.
