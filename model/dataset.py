from pathlib import Path
from typing import Callable, Optional
from PIL import Image
from torch.utils.data import Dataset


ROTATION_ANGLES = [0, 90, 180, 270]


class CrackDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augment = augment

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self.image_files = sorted(
            [
                p
                for p in self.images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        if not self.image_files:
            raise ValueError(f"No image files found in: {self.images_dir}")

        self.samples = []
        missing_masks = []

        for image_path in self.image_files:
            mask_path = self.masks_dir / f"{image_path.stem}.png"

            if mask_path.exists():
                self.samples.append((image_path, mask_path))
            else:
                missing_masks.append((image_path.name, mask_path.name))

        if missing_masks:
            missing_text = "\n".join(
                [f"Image: {img} -> Expected mask: {msk}" for img, msk in missing_masks[:10]]
            )
            raise FileNotFoundError(
                f"Some masks are missing. First examples:\n{missing_text}"
            )

        # Expand samples with rotation angles when augmenting
        if self.augment:
            self.augmented_samples = [
                (image_path, mask_path, angle)
                for image_path, mask_path in self.samples
                for angle in ROTATION_ANGLES
            ]
        else:
            self.augmented_samples = [
                (image_path, mask_path, 0)
                for image_path, mask_path in self.samples
            ]

    def __len__(self) -> int:
        return len(self.augmented_samples)

    def __getitem__(self, idx: int):
        image_path, mask_path, angle = self.augmented_samples[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if angle != 0:
            image = image.rotate(angle, expand=False)
            mask = mask.rotate(angle, expand=False)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
