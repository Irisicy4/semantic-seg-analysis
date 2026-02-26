"""
Dataset -> applicable models (3–4 per dataset). Only run these on the matching dataset.
Config paths are relative to repo root. Checkpoint URLs are downloaded by download_models.sh.
"""
from pathlib import Path
from typing import Dict, List, Tuple

# Repo root (parent of multi_data_semantic_seg)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "mmsegmentation" / "checkpoints"

# Dataset name -> list of (model_name, config_path_from_repo_root, checkpoint_url)
# Configs and URLs from mmsegmentation configs/ READMEs (PSPNet, DeepLabV3+, SegFormer).
DATASET_MODELS: Dict[str, List[Tuple[str, str, str]]] = {
    "ade20k": [
        (
            "pspnet_r50",
            "mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py",
            "https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_80k_ade20k/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth",
        ),
        (
            "pspnet_r101",
            "mmsegmentation/configs/pspnet/pspnet_r101-d8_4xb4-80k_ade20k-512x512.py",
            "https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x512_80k_ade20k/pspnet_r101-d8_512x512_80k_ade20k_20200614_031423-b6e782f0.pth",
        ),
        (
            "deeplabv3plus_r50",
            "mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py",
            "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth",
        ),
        (
            "segformer_mit-b2",
            "mmsegmentation/configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py",
            "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth",
        ),
    ],
    "cityscapes": [
        (
            "pspnet_r50",
            "mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py",
            "https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth",
        ),
        (
            "deeplabv3plus_r50",
            "mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py",
            "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth",
        ),
        (
            "deeplabv3plus_r101",
            "mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.py",
            "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_cityscapes/deeplabv3plus_r101-d8_512x1024_40k_cityscapes_20200605_094614-3769eecf.pth",
        ),
        (
            "segformer_mit-b2",
            "mmsegmentation/configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py",
            "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth",
        ),
    ],
}


def get_checkpoint_path(url: str, checkpoint_dir: Path) -> Path:
    """Local path where the checkpoint will be saved (basename of URL)."""
    return checkpoint_dir / url.split("/")[-1]


def get_models_for_dataset(
    dataset_name: str,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
) -> List[Tuple[str, str, Path]]:
    """
    Returns list of (model_name, config_path_abs, checkpoint_path) for the dataset.
    config_path_abs is absolute path; checkpoint_path may not exist yet.
    """
    if dataset_name not in DATASET_MODELS:
        return []
    out = []
    for name, config_rel, url in DATASET_MODELS[dataset_name]:
        config_abs = (REPO_ROOT / config_rel).resolve()
        ckpt = get_checkpoint_path(url, checkpoint_dir)
        out.append((name, str(config_abs), ckpt))
    return out
