from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.nn import functional as F


class BmpPngDataset(Dataset):
    """
    BMP 이미지 + PNG 마스크용 커스텀 데이터셋.
    지원 구조:
      root/train/images/*.bmp, root/train/masks/*.png
      root/train/*.bmp, root/train/*.png
      root/images/*.bmp, root/masks/*.png
    (선택) classes.txt: 클래스명 1줄 1개
    """

    CLASSES = ["background", "object"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        ignore_label: int = 255,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.ignore_label = ignore_label
        self.resize = resize  # crop size (H, W)

        classes_path = self.root / "classes.txt"
        if classes_path.exists():
            self.CLASSES = self._read_classes(classes_path)
        self.n_classes = len(self.CLASSES)

        split_dir = self.root / split
        split_images = split_dir / "images"
        split_masks = split_dir / "masks"
        root_images = self.root / "images"
        root_masks = self.root / "masks"

        if split_images.exists():
            self.img_dir = split_images
            self.mask_dir = split_masks if split_masks.exists() else split_images
        elif split_dir.exists():
            self.img_dir = split_dir
            self.mask_dir = split_dir
        elif root_images.exists():
            self.img_dir = root_images
            self.mask_dir = root_masks if root_masks.exists() else root_images
        else:
            self.img_dir = self.root
            self.mask_dir = self.root

        self.files = sorted(list(self.img_dir.glob("*.bmp")) + list(self.img_dir.glob("*.BMP")))
        if not self.files:
            raise RuntimeError(f"No BMP images found in {self.img_dir}")

        print(f"[BmpPngDataset] Found {len(self.files)} {split} images")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = self.files[index]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            alt_mask = self.mask_dir / f"{img_path.stem}.PNG"
            if alt_mask.exists():
                mask_path = alt_mask
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path.name}: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        mask_img = Image.open(mask_path)
        mask = torch.from_numpy(np.array(mask_img))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).long()

        if self.resize:
            image, mask = self._crop(image, mask, self.resize)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def _crop(self, image: Tensor, mask: Tensor, size: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        crop_h, crop_w = size
        _, h, w = image.shape
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=self.ignore_label)
            _, h, w = image.shape

        if self.split == "train":
            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()
        else:
            top = max(0, (h - crop_h) // 2)
            left = max(0, (w - crop_w) // 2)

        image = image[:, top : top + crop_h, left : left + crop_w]
        mask = mask[top : top + crop_h, left : left + crop_w]
        return image, mask

    def _read_classes(self, path: Path) -> List[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        classes = [line.strip() for line in lines if line.strip()]
        return classes if classes else self.CLASSES
