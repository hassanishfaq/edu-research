"""Train the four ResNet-50 class-specialist nodes on Google Colab (or the RTX 3090) with the
released thesis code, then download the checkpoints for this demo.

Colab cell 1  (upload Short_Dataset.rar + Zero-Day-Malware-Detection-main.zip to Drive first):
    !pip -q install pyyaml joblib
    !unrar x -o+ "/content/drive/MyDrive/Short_Dataset.rar" /content/data/ > /dev/null
    !unzip -q -o "/content/drive/MyDrive/Zero-Day-Malware-Detection-main.zip" -d /content/
    %cd /content/Zero-Day-Malware-Detection-main
    !python /content/drive/MyDrive/colab_train_nodes.py --data /content/data/Short_Dataset --epochs 10

What this script does
  1. writes a config.yaml for train_node.py with the 4 x 7 family partition of the demo
     (each node gets a data_dir containing symlinks to its seven family folders),
  2. runs  train_node.py  (fine-tunes the backbone head; set --unfreeze to fine-tune everything),
  3. zips outputs/<experiment>/node_*_model.pt (+ nodes.joblib) into fedstack_checkpoints.zip.
Copy the four .pt files into  FedStack-Prototype/models/  and start  python run.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PARTITION = {
    "node_1": ["Adware_ADWARE_Imali_Gen 0", "Adware_ADWARE_InstallRex_Gen2", "Adware_ADWARE_InstMonster_Gen71", "Adware_ADWARE_MultiPlug_Gen73", "Adware_Mplug_LO4", "Gen_Adware_BrowseFox_16", "Gen_Variant_Adware_Mplug"],
    "node_2": ["Gen_Variant_Adware_Graftor", "Gen_Variant_Adware_Kazy", "Gen_Variant_Adware_MultiPlug", "Gen_Variant_Application_Bundler", "Gen_Variant_Application_Kazy", "Email_Worm_W32_Mydoom_gen_A5", "Net_Worm_W32_Allaple_gen_B"],
    "node_3": ["PotentialRisk_PUA_Multiplug_aoa", "PotentialRisk_PUA_Outbrowse_Gen", "PotentialRisk_PUA_Somoto_Gen2", "Trojan_Agent_BDMJ", "Trojan_TR_Agent_nmczf", "Trojan_TR_Crypt_XPACK_Gen", "Trojan_TR_Dropper_Gen"],
    "node_4": ["Trojan_W32_Cryptolocker_C", "Win32_Parite_B", "Win32_Parite_C", "Win32_Ramnit", "Win32_Sality_3", "Win32_Virlock_Gen_1", "Win32_Virtob_Gen_12"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the 28 family sub-folders")
    ap.add_argument("--repo", default=".", help="path of Zero-Day-Malware-Detection-main")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--unfreeze", action="store_true", help="fine-tune the whole backbone (thesis setting) instead of the head only")
    ap.add_argument("--experiment", default="fedstack_demo")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    repo = Path(args.repo).resolve()
    nodes_root = repo / "node_data"
    nodes = []
    for name, fams in PARTITION.items():
        d = nodes_root / name
        d.mkdir(parents=True, exist_ok=True)
        for f in fams:
            src = data / f
            if not src.exists():  # tolerate the folder with the stray space
                alt = [p for p in data.iterdir() if "".join(p.name.split()) == "".join(f.split())]
                src = alt[0] if alt else src
            dst = d / "".join(f.split())
            if not dst.exists():
                try:
                    os.symlink(src, dst, target_is_directory=True)
                except OSError:
                    shutil.copytree(src, dst)
        nodes.append({"name": name, "data_dir": str(d), "include_classes": []})

    cfg = {
        "experiment_name": args.experiment, "private_data_root": str(data), "malimg_data_root": "", "image_size": 224, "grayscale": True,
        "nodes": nodes, "zero_day_holdout_classes": [], "backbone": args.backbone, "pretrained": True, "freeze_backbone": not args.unfreeze,
        "batch_size": args.batch, "epochs": args.epochs, "lr": args.lr, "weight_decay": 1e-4, "num_workers": 2, "seed": 42,
        "node_head": "linear", "stacker": "logreg", "accuracy_weighting": True, "out_dir": str(repo / "outputs"),
    }
    cfg_path = repo / "config_demo.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("config written:", cfg_path)
    subprocess.run([sys.executable, "train_node.py", "--cfg", str(cfg_path)], cwd=repo, check=True)

    out = repo / "outputs" / args.experiment
    zip_base = repo / "fedstack_checkpoints"
    shutil.make_archive(str(zip_base), "zip", out)
    print(f"\nDone. Download {zip_base}.zip and copy node_*_model.pt into FedStack-Prototype/models/")


if __name__ == "__main__":
    main()
