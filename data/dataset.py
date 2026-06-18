"""
UNITY Dataset Module
====================
Supports COCO 2017 and MultiGen-20M datasets for controllable image generation.

Dataset modes
-------------
  coco      — MS-COCO 2017 only (supports all 4 conditions incl. segmentation)
  multigen  — MultiGen-20M only  (canny, depth, scribble; NO segmentation)
  both      — concatenation of the two (segmentation only from COCO)

Condition keys
--------------
  "all"              → use every condition available for the chosen dataset
  "canny"            → Canny edge maps
  "depth_leres"      → Depth maps (LeReS / MiDaS)
  "scribble_pidinet" → Scribble maps (HED / PidiNet)
  "segmentation"     → Semantic segmentation (COCO only)

Expected directory layout
--------------------------
data/
├── coco/
│   ├── train2017/                      # JPEG images
│   ├── annotations/
│   │   └── captions_train2017.json
│   ├── canny/                          # .png  (same stem as image)
│   ├── depth/
│   ├── scribble/
│   └── segmentation/
└── multigen20m/
    ├── annotations/
    │   └── train.json                  # list of {"image": str, "caption": str}
    ├── source/                         # original images
    ├── canny/
    ├── depth/
    └── hed/                            # scribble maps (HED)
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All conditions defined by the project (in fixed order)
ALL_COCO_CONDITIONS: List[str] = ["canny", "depth_leres", "scribble_pidinet", "segmentation"]
ALL_MULTIGEN_CONDITIONS: List[str] = ["canny", "depth_leres", "scribble_pidinet"]

# Subfolder names within each dataset root
COCO_COND_DIRS = {
    "canny":            "canny",
    "depth_leres":      "depth",
    "scribble_pidinet": "scribble",
    "segmentation":     "segmentation",
}

MULTIGEN_COND_DIRS = {
    "canny":            "canny",
    "depth_leres":      "depth",
    "scribble_pidinet": "hed",
}

# Number of channels per condition (always RGB)
CHANNELS_PER_COND = 3
# Total channels when ALL 4 conditions are stacked (used in Phase 1)
TOTAL_CHANNELS = len(ALL_COCO_CONDITIONS) * CHANNELS_PER_COND  # 12


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def _image_transform(resolution: int) -> transforms.Compose:
    """Resize + normalise to [-1, 1] for target images."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution),
                          interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _condition_transform(resolution: int) -> transforms.Compose:
    """Resize + scale to [0, 1] for condition maps."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution),
                          interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Text encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_sd15(caption: str, tokenizer, text_encoder, max_length: int = 77) -> torch.Tensor:
    """Return last-hidden-state embeddings for SD 1.5.  Shape: [77, 768]"""
    ids = tokenizer(
        caption,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids
    text_encoder.eval()
    return text_encoder(ids)[0].squeeze(0)   # [77, 768]


@torch.no_grad()
def _encode_sdxl(
    caption: str,
    tokenizers: list,
    text_encoders: list,
    max_length: int = 77,
):
    """
    Return (prompt_embeds, pooled_embeds) for SDXL.

    prompt_embeds  : [77, 2048]  — penultimate hidden states of both encoders concat'd
    pooled_embeds  : [1280]      — pooled output of text_encoder_2
    """
    hidden_states = []
    pooled = None
    for tok, enc in zip(tokenizers, text_encoders):
        ids = tok(
            caption,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        enc.eval()
        out = enc(ids, output_hidden_states=True)
        # Use the penultimate layer (standard for SDXL)
        hidden_states.append(out.hidden_states[-2].squeeze(0))   # [77, D]
        # text_embeds is the pooled projection (only present in CLIPTextModelWithProjection)
        if hasattr(out, "text_embeds") and out.text_embeds is not None:
            pooled = out.text_embeds.squeeze(0)                  # [1280]

    return torch.cat(hidden_states, dim=-1), pooled              # [77, 2048], [1280]


# ---------------------------------------------------------------------------
# Shared collate function
# ---------------------------------------------------------------------------

def _collate(batch: list) -> dict:
    pixel_values              = torch.stack([b["pixel_values"] for b in batch])
    conditioning_pixel_values = torch.stack([b["conditioning_pixel_values"] for b in batch])
    prompt_ids                = torch.stack([b["prompt_ids"] for b in batch])

    # unet_added_conditions: stack each sub-tensor separately
    unet_added_conditions: dict = {}
    if batch[0]["unet_added_conditions"]:
        for key in batch[0]["unet_added_conditions"]:
            unet_added_conditions[key] = torch.stack(
                [b["unet_added_conditions"][key] for b in batch]
            )

    return {
        "pixel_values":              pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids":                prompt_ids,
        "unet_added_conditions":     unet_added_conditions,
    }


# ---------------------------------------------------------------------------
# COCO Dataset
# ---------------------------------------------------------------------------

class COCODataset(Dataset):
    """
    MS-COCO 2017 paired dataset.

    Supports all four conditions: canny, depth_leres, scribble_pidinet, segmentation.
    If a condition map file is missing for a given image, a black (zero) tensor is
    returned for that condition — this allows the dataset to be used even when
    condition maps have not been generated for every image.

    Args:
        root        : Path to the dataset root that contains the ``coco/`` folder.
        tokenizer   : A single tokenizer (SD 1.5) or list of two (SDXL).
        text_encoder: A single text encoder (SD 1.5) or list of two (SDXL).
        conditions  : ``"all"`` or one of the four condition key strings.
        resolution  : Square output resolution in pixels.
        is_sdxl     : Set True when using SDXL dual-encoder setup.
    """

    def __init__(
        self,
        root: str,
        tokenizer,
        text_encoder,
        conditions: str,
        resolution: int = 512,
        is_sdxl: bool = False,
    ):
        self.root        = os.path.join(root, "coco")
        self.img_dir     = os.path.join(self.root, "train2017")
        self.resolution  = resolution
        self.is_sdxl     = is_sdxl
        self.tokenizer   = tokenizer
        self.text_encoder = text_encoder

        self.conditions: List[str] = (
            ALL_COCO_CONDITIONS if conditions == "all" else [conditions]
        )

        self.img_transform  = _image_transform(resolution)
        self.cond_transform = _condition_transform(resolution)

        # ── Load captions ──────────────────────────────────────────────────
        ann_path = os.path.join(
            self.root, "annotations", "captions_train2017.json"
        )
        with open(ann_path) as f:
            data = json.load(f)

        id2fn = {img["id"]: img["file_name"] for img in data["images"]}
        self.samples = [
            (id2fn[ann["image_id"]], ann["caption"])
            for ann in data["annotations"]
        ]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _zero(self) -> torch.Tensor:
        return torch.zeros(3, self.resolution, self.resolution)

    def _load_cond(self, stem: str, condition: str) -> torch.Tensor:
        cdir = COCO_COND_DIRS[condition]
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(self.root, cdir, stem + ext)
            if os.path.exists(path):
                return self.cond_transform(Image.open(path).convert("RGB"))
        return self._zero()

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        filename, caption = self.samples[idx]
        stem = os.path.splitext(filename)[0]

        # Target image
        pixel_values = self.img_transform(
            Image.open(os.path.join(self.img_dir, filename)).convert("RGB")
        )

        # Condition maps stacked along channel dimension → [C*3, H, W]
        cond_maps = [self._load_cond(stem, c) for c in self.conditions]
        conditioning_pixel_values = torch.cat(cond_maps, dim=0)

        # Text embeddings
        if self.is_sdxl:
            prompt_ids, pooled = _encode_sdxl(caption, self.tokenizer, self.text_encoder)
            unet_added_conditions = {
                "text_embeds": pooled,
                "time_ids": torch.tensor(
                    [self.resolution, self.resolution, 0, 0,
                     self.resolution, self.resolution],
                    dtype=torch.float32,
                ),
            }
        else:
            prompt_ids = _encode_sd15(caption, self.tokenizer[0], self.text_encoder[0])
            unet_added_conditions = {}

        return {
            "pixel_values":              pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "prompt_ids":                prompt_ids,
            "unet_added_conditions":     unet_added_conditions,
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        return _collate(batch)


# ---------------------------------------------------------------------------
# MultiGen-20M Dataset
# ---------------------------------------------------------------------------

class MultiGenDataset(Dataset):
    """
    MultiGen-20M paired dataset.

    .. important::
        MultiGen-20M does **not** include segmentation maps.
        Supported conditions: ``canny``, ``depth_leres``, ``scribble_pidinet``.
        When ``conditions="all"`` is requested, the segmentation channel slot is
        filled with zeros so the tensor shape matches COCO (12 channels = 4 × 3),
        allowing both datasets to be mixed in the same batch during Phase 1.

    Annotation format
    -----------------
    ``annotations/train.json`` is expected to be a JSON array where each entry is::

        {"image": "<filename>", "caption": "<text>"}

    Args:
        root        : Path to the dataset root that contains ``multigen20m/``.
        tokenizer   : A single tokenizer (SD 1.5) or list of two (SDXL).
        text_encoder: A single text encoder (SD 1.5) or list of two (SDXL).
        conditions  : ``"all"`` or one of ``canny``, ``depth_leres``,
                      ``scribble_pidinet`` (segmentation raises an error).
        resolution  : Square output resolution in pixels.
        is_sdxl     : Set True when using SDXL dual-encoder setup.
    """

    def __init__(
        self,
        root: str,
        tokenizer,
        text_encoder,
        conditions: str,
        resolution: int = 512,
        is_sdxl: bool = False,
    ):
        if conditions == "segmentation":
            raise ValueError(
                "MultiGen-20M does not contain segmentation maps. "
                "Use COCO (dataset_mode='coco') for segmentation fine-tuning."
            )

        self.root         = os.path.join(root, "multigen20m")
        self.resolution   = resolution
        self.is_sdxl      = is_sdxl
        self.tokenizer    = tokenizer
        self.text_encoder = text_encoder

        # When "all" is requested we load the 3 available conditions and then
        # zero-pad the segmentation slot so the output is always 12 channels.
        if conditions == "all":
            self.conditions       = ALL_MULTIGEN_CONDITIONS
            self.pad_segmentation = True
        else:
            self.conditions       = [conditions]
            self.pad_segmentation = False

        self.img_transform  = _image_transform(resolution)
        self.cond_transform = _condition_transform(resolution)

        ann_path = os.path.join(self.root, "annotations", "train.json")
        with open(ann_path) as f:
            self.samples = json.load(f)   # list of {"image": str, "caption": str}

    # ── Helpers ────────────────────────────────────────────────────────────

    def _zero(self) -> torch.Tensor:
        return torch.zeros(3, self.resolution, self.resolution)

    def _load_cond(self, stem: str, condition: str) -> torch.Tensor:
        cdir = MULTIGEN_COND_DIRS[condition]
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(self.root, cdir, stem + ext)
            if os.path.exists(path):
                return self.cond_transform(Image.open(path).convert("RGB"))
        return self._zero()

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        entry    = self.samples[idx]
        filename = entry["image"]
        caption  = entry["caption"]
        stem     = os.path.splitext(os.path.basename(filename))[0]

        # Target image
        img_path = os.path.join(self.root, "source", filename)
        pixel_values = self.img_transform(Image.open(img_path).convert("RGB"))

        # Condition maps
        cond_maps = [self._load_cond(stem, c) for c in self.conditions]
        if self.pad_segmentation:
            # Append a black tensor in the segmentation slot
            cond_maps.append(self._zero())
        conditioning_pixel_values = torch.cat(cond_maps, dim=0)

        # Text embeddings
        if self.is_sdxl:
            prompt_ids, pooled = _encode_sdxl(caption, self.tokenizer, self.text_encoder)
            unet_added_conditions = {
                "text_embeds": pooled,
                "time_ids": torch.tensor(
                    [self.resolution, self.resolution, 0, 0,
                     self.resolution, self.resolution],
                    dtype=torch.float32,
                ),
            }
        else:
            prompt_ids = _encode_sd15(caption, self.tokenizer[0], self.text_encoder[0])
            unet_added_conditions = {}

        return {
            "pixel_values":              pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "prompt_ids":                prompt_ids,
            "unet_added_conditions":     unet_added_conditions,
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        return _collate(batch)


# ---------------------------------------------------------------------------
# Combined Dataset wrapper
# ---------------------------------------------------------------------------

class _CombinedDataset(Dataset):
    """Thin wrapper around ConcatDataset that exposes a unified collate_fn."""

    def __init__(self, datasets: list):
        self._inner = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        return self._inner[idx]

    @staticmethod
    def collate_fn(batch: list) -> dict:
        return _collate(batch)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_train_dataset(
    root_path: str,
    tokenizer,
    text_encoder,
    conditions: str,
    resolution: int = 512,
    model_variant: Optional[str] = None,
    dataset_mode: str = "coco",
) -> Dataset:
    """
    Create a training dataset for UNITY.

    Parameters
    ----------
    root_path:
        Root directory that contains ``coco/`` and/or ``multigen20m/`` sub-folders.
    tokenizer:
        Single tokenizer for SD 1.5, or ``[tokenizer_1, tokenizer_2]`` for SDXL.
    text_encoder:
        Single text encoder for SD 1.5, or ``[encoder_1, encoder_2]`` for SDXL.
    conditions:
        ``"all"`` | ``"canny"`` | ``"depth_leres"`` | ``"scribble_pidinet"``
        | ``"segmentation"``.
        Note: ``"segmentation"`` and ``"all"`` in MultiGen-20M mode will raise or
        pad respectively — see class docstrings for details.
    resolution:
        Square crop/resize target in pixels (default 512).
    model_variant:
        Ignored; kept for backward compatibility.
    dataset_mode:
        ``"coco"`` | ``"multigen"`` | ``"both"``.

    Returns
    -------
    A :class:`torch.utils.data.Dataset` with a ``collate_fn`` static method.
    """
    # Detect SDXL by checking for a pair of tokenizers
    is_sdxl: bool = isinstance(tokenizer, (list, tuple)) and len(tokenizer) == 2

    shared = dict(
        root         = root_path,
        tokenizer    = tokenizer,
        text_encoder = text_encoder,
        conditions   = conditions,
        resolution   = resolution,
        is_sdxl      = is_sdxl,
    )

    if dataset_mode == "coco":
        return COCODataset(**shared)

    elif dataset_mode == "multigen":
        return MultiGenDataset(**shared)

    elif dataset_mode == "both":
        coco_ds = COCODataset(**shared)
        if conditions == "segmentation":
            # Segmentation not available in MultiGen-20M — fall back to COCO only
            return coco_ds
        multigen_ds = MultiGenDataset(**shared)
        return _CombinedDataset([coco_ds, multigen_ds])

    else:
        raise ValueError(
            f"Unknown dataset_mode '{dataset_mode}'. "
            "Valid options: 'coco', 'multigen', 'both'."
        )
