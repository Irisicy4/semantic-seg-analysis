"""ADE20K dataset loader for semantic segmentation inference.

Expects data root: .../ADE20K_2021_17_01_val with structure:
  - images/ADE/validation/.../  containing  {stem}.jpg, {stem}_seg.png, {stem}.json
  - objects.txt  (optional, for id -> class name mapping)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# Default ADE20K 150 classes (mmseg order; indices 0..149 after reduce_zero_label)
# Used to map full-taxonomy name from objects.txt to 150-class index.
ADE20K_CLASSES = (
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
    'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
    'person', 'earth', 'door', 'table', 'mountain', 'plant',
    'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
    'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
    'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
    'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
    'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
    'screen door', 'stairway', 'river', 'bridge', 'bookcase',
    'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
    'bench', 'countertop', 'stove', 'palm', 'kitchen island',
    'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
    'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
    'chandelier', 'awning', 'streetlight', 'booth',
    'television receiver', 'airplane', 'dirt track', 'apparel',
    'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
    'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
    'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
    'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
    'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
    'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
    'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
    'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
    'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag',
)


def _load_ade20k_objects_txt(objects_path: Path) -> Dict[int, str]:
    """Parse objects.txt; return mapping name_ndx -> ADE name (first of ADE names column, index 4)."""
    id_to_name: Dict[int, str] = {}
    with open(objects_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            try:
                name_ndx = int(parts[1])
                ade_names = parts[4].strip()
                id_to_name[name_ndx] = ade_names if ade_names else str(name_ndx)
            except (ValueError, IndexError):
                continue
    return id_to_name


def _build_name_ndx_to_150(id_to_name: Dict[int, str]) -> Dict[int, int]:
    """Map full-taxonomy name_ndx -> 0..149 using objects.txt names and ADE20K 150-class list."""
    class_name_to_idx = {c.strip().lower(): i for i, c in enumerate(ADE20K_CLASSES)}
    out: Dict[int, int] = {}
    for name_ndx, ade_names_str in id_to_name.items():
        for part in ade_names_str.split(','):
            key = part.strip().lower()
            if key in class_name_to_idx:
                out[name_ndx] = class_name_to_idx[key]
                break
    return out


class ADE20KDataset:
    """Load ADE20K (ADE20K_2021_17_01_val) images and ground truth for inference."""

    def __init__(
        self,
        data_root: str,
        split: str = 'validation',
        reduce_zero_label: bool = True,
        ignore_index: int = 255,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index
        self.max_samples = max_samples

        self.images_dir = self.data_root / 'images' / 'ADE' / split
        if not self.images_dir.exists():
            self.images_dir = self.data_root / 'images'
        objects_path = self.data_root / 'objects.txt'
        self.id_to_name = _load_ade20k_objects_txt(objects_path) if objects_path.exists() else {}
        self._name_ndx_to_150 = _build_name_ndx_to_150(self.id_to_name) if self.id_to_name else {}
        self._samples: List[Tuple[Path, Path]] = []

    def _collect_samples(self) -> List[Tuple[Path, Path]]:
        samples = []
        for jpg_path in self.images_dir.rglob('*.jpg'):
            stem = jpg_path.stem
            seg_path = jpg_path.parent / f'{stem}_seg.png'
            if seg_path.exists():
                samples.append((jpg_path, seg_path))
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
        """Return ordered class names (for standard 0..N-1 or 1..N after reduce_zero_label)."""
        if self.id_to_name:
            max_id = max(self.id_to_name.keys())
            out = ['unlabeled'] * (max_id + 1)
            for i, name in self.id_to_name.items():
                out[i] = name
            return out
        return ['unlabeled'] + list(ADE20K_CLASSES)

    def load_image(self, image_path: Path) -> np.ndarray:
        img = Image.open(image_path).convert('RGB')
        return np.array(img)

    def load_gt(self, seg_path: Path) -> np.ndarray:
        """
        Load ground truth from ADE20K _seg.png.
        ADE20K 2021: RGB-encoded — R,G encode class (class_raw = (R/10)*256 + G, full taxonomy);
        we map class_raw to 0..149 via objects.txt + ADE20K 150-class list.
        If objects.txt is missing, fall back to single-channel (legacy) and treat as 0=ignore, 1..150->0..149.
        """
        seg = np.array(Image.open(seg_path))
        if seg.ndim == 3 and self._name_ndx_to_150:
            # ADE20K 2021 RGB encoding (see CSAILVision/ADE20K utils/utils_ade20k.py)
            R = np.asarray(seg[:, :, 0], dtype=np.int32)
            G = np.asarray(seg[:, :, 1], dtype=np.int32)
            class_raw = (R // 10) * 256 + G
            out = np.full(class_raw.shape, self.ignore_index, dtype=np.int32)
            for name_ndx, idx_150 in self._name_ndx_to_150.items():
                out[class_raw == name_ndx] = idx_150
            return out
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        if self.reduce_zero_label:
            out = np.full_like(seg, self.ignore_index)
            mask = seg > 0
            out[mask] = seg[mask].astype(np.int32) - 1
            return out
        return seg.astype(np.int32)

    def __getitem__(self, idx: int) -> dict:
        image_path, seg_path = self.samples[idx]
        image = self.load_image(image_path)
        gt = self.load_gt(seg_path)
        stem = image_path.stem
        return {
            'image': image,
            'gt': gt,
            'image_path': str(image_path),
            'seg_path': str(seg_path),
            'sample_id': stem,
            'dataset': 'ade20k',
            'height': image.shape[0],
            'width': image.shape[1],
        }
