import argparse
from pathlib import Path
from typing import Optional

import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from config import load_config
from datasets import BmpPngDataset
from metrics import SegMetrics
from models import build_model


def _parse_resize(value):
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("TRAIN.IMAGE_SIZE는 정수 또는 [H, W] 형식(크롭 사이즈)이어야 합니다.")


def evaluate(cfg: dict, split: Optional[str] = None):
    device = torch.device(cfg["DEVICE"])
    eval_cfg = cfg["EVAL"]
    dataset_cfg, model_cfg = cfg["DATASET"], cfg["MODEL"]

    split = split or eval_cfg.get("SPLIT", "val")
    resize = _parse_resize(cfg["TRAIN"].get("IMAGE_SIZE"))
    dataset = BmpPngDataset(dataset_cfg["ROOT"], split, None, dataset_cfg["IGNORE_LABEL"], resize=resize)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)

    model = build_model(model_cfg, dataset.n_classes).to(device)
    model_path = eval_cfg.get("MODEL_PATH")
    if not model_path:
        head_name = model_cfg.get("HEAD") or model_cfg.get("NAME")
        model_path = Path(cfg["SAVE_DIR"]) / f"{head_name}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth"
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    metrics = SegMetrics(dataset.n_classes, dataset.ignore_label, device)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, mode="tensor")
            metrics.update(logits.softmax(dim=1), labels)

    ious, miou = metrics.compute_iou()
    dice, mdice = metrics.compute_dice()

    table = {
        "Class": list(dataset.CLASSES) + ["Mean"],
        "IoU": ious + [miou],
        "Dice": dice + [mdice],
    }
    print(tabulate(table, headers="keys"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    evaluate(cfg, split=args.split)


if __name__ == "__main__":
    main()
