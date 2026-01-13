import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


def _resolve_dirs(
    root: Path, images_dir: Optional[str], masks_dir: Optional[str]
) -> Tuple[Path, Path]:
    if images_dir:
        img_dir = Path(images_dir)
        mask_dir = Path(masks_dir) if masks_dir else Path(images_dir)
        return img_dir, mask_dir

    root_images = root / "images"
    root_masks = root / "masks"
    if root_images.exists():
        return root_images, root_masks if root_masks.exists() else root_images
    return root, root


def _collect_pairs(img_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    images = sorted(list(img_dir.glob("*.bmp")) + list(img_dir.glob("*.BMP")))
    if not images:
        raise RuntimeError(f"BMP 이미지를 찾을 수 없습니다: {img_dir}")

    pairs = []
    for img_path in images:
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            alt = mask_dir / f"{img_path.stem}.PNG"
            if alt.exists():
                mask_path = alt
        if not mask_path.exists():
            raise FileNotFoundError(f"마스크가 없습니다: {img_path.name}")
        pairs.append((img_path, mask_path))
    return pairs


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"이미 존재함: {dst}")
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}")


def split_dataset(
    input_root: str,
    output_root: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    mode: str,
    overwrite: bool,
    images_dir: Optional[str] = None,
    masks_dir: Optional[str] = None,
) -> None:
    root = Path(input_root)
    out_root = Path(output_root)

    if not root.exists():
        raise FileNotFoundError(f"입력 경로가 없습니다: {root}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio는 1.0보다 작아야 합니다.")

    img_dir, mask_dir = _resolve_dirs(root, images_dir, masks_dir)
    pairs = _collect_pairs(img_dir, mask_dir)

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_train, n_val, n_test = _split_indices(len(pairs), train_ratio, val_ratio)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split_name, items in splits.items():
        img_out = out_root / split_name / "images"
        mask_out = out_root / split_name / "masks"
        _ensure_dir(img_out)
        _ensure_dir(mask_out)

        for img_path, mask_path in items:
            _copy_file(img_path, img_out / img_path.name, mode, overwrite)
            _copy_file(mask_path, mask_out / mask_path.name, mode, overwrite)

    classes_path = root / "classes.txt"
    if classes_path.exists():
        dst = out_root / "classes.txt"
        if not dst.exists() or overwrite:
            shutil.copy2(classes_path, dst)

    print(
        f"[Split] total={len(pairs)} train={n_train} val={n_val} test={n_test} -> {out_root}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--masks-dir", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        mode=args.mode,
        overwrite=args.overwrite,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
    )


if __name__ == "__main__":
    main()
