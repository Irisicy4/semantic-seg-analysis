"""Cityscapes dataset loader for semantic segmentation inference.

Expects data root (e.g. multi_data_semantic_seg/data/gtFine_trainvaltest) with:
  - gtFine/{split}/{city}/{id}_gtFine_labelIds.png
  - leftImg8bit/{split}/{city}/{id}_leftImg8bit.png  (required; no fallback)
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# Cityscapes 19 train classes (trainId 0-18); 255 = ignore
CITYSCAPES_CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle',
)

# Map labelId to trainId (Cityscapes convention)
LABEL_ID_TO_TRAIN_ID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255,
    17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18,
}
for i in range(34, 256):
    LABEL_ID_TO_TRAIN_ID.setdefault(i, 255)


def label_ids_to_train_ids(seg: np.ndarray) -> np.ndarray:
    out = np.full_like(seg, 255)
    for lid, tid in LABEL_ID_TO_TRAIN_ID.items():
        out[seg == lid] = tid
    return out


class CityscapesDataset:
    """Load Cityscapes (gtFine) images and ground truth for inference."""

    def __init__(
        self,
        data_root: str,
        image_root: Optional[str] = None,
        split: str = 'val',
        use_train_id: bool = True,
        ignore_index: int = 255,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.image_root = Path(image_root) if image_root else self.data_root
        self.split = split
        self.use_train_id = use_train_id
        self.ignore_index = ignore_index
        self.max_samples = max_samples

        self.gt_dir = self.data_root / 'gtFine' / split
        self.img_dir = self.image_root / 'leftImg8bit' / split
        self._samples: List[Tuple[Path, Path]] = []

    def _collect_samples(self) -> List[Tuple[Path, Path]]:
        samples = []
        for gt_path in self.gt_dir.rglob('*_gtFine_labelIds.png'):
            city = gt_path.parent.name
            stem = gt_path.stem.replace('_gtFine_labelIds', '')
            img_path = self.img_dir / city / f'{stem}_leftImg8bit.png'
            if img_path.exists():
                samples.append((img_path, gt_path))
        samples.sort(key=lambda x: str(x[0]))
        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    @property
    def samples(self) -> List[Tuple[Path, Path]]:
        if not self._samples:
            self._samples = self._collect_samples()
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_class_names(self) -> List[str]:
        return list(CITYSCAPES_CLASSES)

    def load_image(self, image_path: Path) -> np.ndarray:
        img = Image.open(image_path).convert('RGB')
        return np.array(img)

    def load_gt(self, seg_path: Path) -> np.ndarray:
        seg = np.array(Image.open(seg_path))
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        if self.use_train_id:
            seg = label_ids_to_train_ids(seg.astype(np.int32))
        return seg.astype(np.int32)

    def __getitem__(self, idx: int) -> dict:
        image_path, seg_path = self.samples[idx]
        image = self.load_image(image_path)
        gt = self.load_gt(seg_path)
        stem = image_path.stem.replace('_leftImg8bit', '')
        return {
            'image': image,
            'gt': gt,
            'image_path': str(image_path),
            'seg_path': str(seg_path),
            'sample_id': stem,
            'dataset': 'cityscapes',
            'height': image.shape[0],
            'width': image.shape[1],
        }
