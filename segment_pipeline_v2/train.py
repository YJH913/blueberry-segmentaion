import argparse
import time
from pathlib import Path

import torch
from tabulate import tabulate
from contextlib import nullcontext
try:
    from torch.amp import GradScaler, autocast
    _AMP_CONTEXT = autocast
    _AMP_REQUIRES_DEVICE_TYPE = True
except Exception:
    try:
        from torch.cuda.amp import GradScaler, autocast
        _AMP_CONTEXT = autocast
        _AMP_REQUIRES_DEVICE_TYPE = False
    except Exception:
        GradScaler = None
        _AMP_CONTEXT = nullcontext
        _AMP_REQUIRES_DEVICE_TYPE = False
from torch.utils.data import DataLoader, RandomSampler

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


def train(cfg: dict) -> Path:
    device = torch.device(cfg["DEVICE"])
    train_cfg, eval_cfg = cfg["TRAIN"], cfg["EVAL"]
    dataset_cfg, model_cfg = cfg["DATASET"], cfg["MODEL"]

    resize = _parse_resize(train_cfg.get("IMAGE_SIZE"))
    trainset = BmpPngDataset(dataset_cfg["ROOT"], "train", None, dataset_cfg["IGNORE_LABEL"], resize=resize)
    valset = BmpPngDataset(dataset_cfg["ROOT"], "val", None, dataset_cfg["IGNORE_LABEL"], resize=resize)

    model = build_model(model_cfg, trainset.n_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset_cfg["IGNORE_LABEL"])

    sampler = RandomSampler(trainset)
    micro_batch_size = train_cfg.get("MICRO_BATCH_SIZE", train_cfg["BATCH_SIZE"])
    grad_accum_steps = max(1, int(train_cfg.get("GRAD_ACCUM_STEPS", 1)))
    trainloader = DataLoader(
        trainset,
        batch_size=micro_batch_size,
        num_workers=train_cfg.get("NUM_WORKERS", 4),
        drop_last=True,
        pin_memory=True,
        sampler=sampler,
    )
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)

    opt_cfg = cfg["OPTIMIZER"]
    opt_type = str(opt_cfg.get("TYPE", "AdamW")).lower()
    lr = opt_cfg["LR"]
    weight_decay = opt_cfg.get("WEIGHT_DECAY", 0.0)
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_cfg.get("MOMENTUM", 0.9),
            weight_decay=weight_decay,
            nesterov=bool(opt_cfg.get("NESTEROV", False)),
        )
    else:
        raise ValueError(f"지원하지 않는 OPTIMIZER.TYPE입니다: {opt_cfg.get('TYPE')}")
    scaler = GradScaler(enabled=train_cfg.get("AMP", False)) if GradScaler else None

    iters_per_epoch = max(1, len(trainset) // micro_batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (1 - step / (train_cfg["EPOCHS"] * iters_per_epoch)) ** cfg["SCHEDULER"]["POWER"],
    )

    save_dir = Path(cfg["SAVE_DIR"])
    save_dir.mkdir(exist_ok=True)
    head_name = model_cfg.get("HEAD") or model_cfg.get("NAME")
    save_path = save_dir / f"{head_name}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth"

    for epoch in range(train_cfg["EPOCHS"]):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (img, lbl) in enumerate(trainloader, start=1):
            img = img.to(device)
            lbl = lbl.to(device)

            amp_enabled = train_cfg.get("AMP", False) and _AMP_CONTEXT is not nullcontext
            if _AMP_CONTEXT is not nullcontext and _AMP_REQUIRES_DEVICE_TYPE:
                amp_ctx = _AMP_CONTEXT(device_type=device.type, enabled=amp_enabled)
            elif _AMP_CONTEXT is not nullcontext:
                amp_ctx = _AMP_CONTEXT(enabled=amp_enabled)
            else:
                amp_ctx = nullcontext()
            with amp_ctx:
                logits = model(img)
                loss = loss_fn(logits, lbl) / grad_accum_steps

            epoch_loss += loss.item() * grad_accum_steps
            epoch_steps += 1

            if scaler:
                scaler.scale(loss).backward()
                if step_idx % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if step_idx % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if step_idx % grad_accum_steps == 0:
                scheduler.step()
        if step_idx % grad_accum_steps != 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        if epoch_steps:
            avg_loss = epoch_loss / epoch_steps
            print(f"Epoch [{epoch + 1}/{train_cfg['EPOCHS']}], Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    metrics = SegMetrics(valset.n_classes, valset.ignore_label, device)
    model.eval()
    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            metrics.update(logits.softmax(dim=1), labels)
    ious, miou = metrics.compute_iou()
    dice, mdice = metrics.compute_dice()

    table = {
        "Class": list(valset.CLASSES) + ["Mean"],
        "IoU": ious + [miou],
        "Dice": dice + [mdice],
    }
    print(tabulate(table, headers="keys"))

    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    start = time.time()
    train(cfg)
    end = time.time()
    print(f"Total time: {end - start:.1f}s")


if __name__ == "__main__":
    main()
