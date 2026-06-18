# UNITY: Universal Condition Adapter for Diffusion Models

> **ECCV 2026** — Official PyTorch Implementation

**[Aryan Das](https://aryan-das.netlify.app/)<sup>1</sup>** &nbsp;·&nbsp;
**Koushik Biswas<sup>2</sup>** &nbsp;·&nbsp;
**Moloud Abdar<sup>3</sup>** &nbsp;·&nbsp;
**Vinay Kumar Verma<sup>4</sup>**

UNITY is a universal adapter framework that enables controllable image generation from multiple spatial conditioning signals (canny edges, depth maps, scribbles, segmentation maps) using a single unified model. It builds on top of frozen Stable Diffusion 1.5 or SDXL backbones and follows a two-phase training curriculum.

---

## Repository Structure

```
UNITY/
├── data/
│   ├── __init__.py
│   └── dataset.py            # Dataset classes & get_train_dataset factory
├── models/
│   ├── __init__.py
│   ├── adapter.py            # UNITY adapter (UNITY_Config, UNITY, MAFNet, ...)
│   ├── pipeline.py           # SD 1.5 inference pipeline
│   └── pipeline_sdxl.py      # SDXL inference pipeline
├── scripts/
│   ├── train_p1_sd15.py      # Phase 1 training — SD 1.5
│   ├── train_p1_sdxl.py      # Phase 1 training — SDXL
│   ├── train_p2_sd15.py      # Phase 2 training — SD 1.5
│   └── train_p2_sdxl.py      # Phase 2 training — SDXL
├── run_p1_sd15.sh            # Launch Phase 1 — SD 1.5
├── run_p1_sdxl.sh            # Launch Phase 1 — SDXL
├── run_p2_sd15.sh            # Launch Phase 2 — SD 1.5
├── run_p2_sdxl.sh            # Launch Phase 2 — SDXL
├── utils.py                  # Shared argument parser
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
git clone https://github.com/<your-org>/UNITY.git
cd UNITY

conda create -n unity python=3.10 -y
conda activate unity

# PyTorch — adjust the CUDA version tag as needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

---

## Dataset Preparation

UNITY uses two datasets. The table below shows which conditions each one provides.

| Dataset | canny | depth | scribble | segmentation |
|---|:---:|:---:|:---:|:---:|
| **MS-COCO 2017** | ✓ | ✓ | ✓ | ✓ |
| **MultiGen-20M** | ✓ | ✓ | ✓ | ✗ |

> **MultiGen-20M does not include segmentation maps.**  
> When training with `--conditions=all` on MultiGen, the segmentation channel is automatically zero-padded so the 12-channel tensor shape is consistent with COCO.  
> For segmentation-specific fine-tuning (Phase 2) always use `--dataset_mode=coco`.

---

### Expected Directory Layout

Place (or symlink) the datasets inside the `data/` folder:

```
data/
├── coco/
│   ├── train2017/                   # JPEG images
│   ├── annotations/
│   │   └── captions_train2017.json
│   ├── canny/                       # .png, same stem as image
│   ├── depth/
│   ├── scribble/
│   └── segmentation/
└── multigen20m/
    ├── annotations/
    │   └── train.json               # [{"image": "<filename>", "caption": "..."}, ...]
    ├── source/                      # original images
    ├── canny/
    ├── depth/
    └── hed/                         # scribble (HED detector output)
```

---

### Step 1 — Download Images

**COCO 2017**

```bash
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip
cd ../..
```

**MultiGen-20M** (via HuggingFace Hub)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='showlab/MultiGen-20M', repo_type='dataset', local_dir='data/multigen20m')
"
```

---

### Step 2 — Generate Condition Maps (COCO)

MultiGen-20M ships with pre-computed condition maps. For COCO you need to generate them. Install the annotation tools:

```bash
pip install controlnet-aux
# For segmentation: install OneFormer or Mask2Former separately
```

Then run your preferred batch script. A minimal example for a single image:

```python
from controlnet_aux import CannyDetector, MidasDetector, HEDdetector
from PIL import Image

img = Image.open("data/coco/train2017/000000001000.jpg")

CannyDetector()(img).save("data/coco/canny/000000001000.png")
MidasDetector.from_pretrained("lllyasviel/Annotators")(img).save("data/coco/depth/000000001000.png")
HEDdetector.from_pretrained("lllyasviel/Annotators")(img, scribble=True).save("data/coco/scribble/000000001000.png")
# Segmentation: use OneFormer / Mask2Former and save to data/coco/segmentation/
```

Scale this up to all ~118k COCO training images before starting training.

---

### Dataset Module

The full implementation is in [`data/dataset.py`](data/dataset.py).  
It provides `COCODataset`, `MultiGenDataset`, and a `get_train_dataset` factory that accepts the following arguments:

| Argument | Type | Description |
|---|---|---|
| `root_path` | `str` | Root directory containing `coco/` and/or `multigen20m/` |
| `tokenizer` | tokenizer or `[tok1, tok2]` | SD 1.5 tokenizer, or pair for SDXL |
| `text_encoder` | encoder or `[enc1, enc2]` | SD 1.5 text encoder, or pair for SDXL |
| `conditions` | `str` | `"all"` \| `"canny"` \| `"depth_leres"` \| `"scribble_pidinet"` \| `"segmentation"` |
| `resolution` | `int` | Image resolution (default 512) |
| `dataset_mode` | `str` | `"coco"` \| `"multigen"` \| `"both"` |

Each `__getitem__` returns:

```python
{
    "pixel_values":              torch.Tensor,  # [3, H, W]  target image, normalised to [-1, 1]
    "conditioning_pixel_values": torch.Tensor,  # [C*3, H, W] stacked condition maps (C=4 Phase1, C=1 Phase2)
    "prompt_ids":                torch.Tensor,  # [77, D]    CLIP text embeddings
    "unet_added_conditions":     dict,          # SDXL only: {"text_embeds": ..., "time_ids": ...}
}
```

---

## Two-Phase Training

| | Phase 1 | Phase 2 |
|---|---|---|
| **Goal** | Learn universal multi-condition features | Fine-tune per condition |
| **`--conditions`** | `all` | `canny` / `depth_leres` / `scribble_pidinet` / `segmentation` |
| **Adapter input channels** | 12 (4 × 3ch) | 3 (1 × 3ch) |
| **Initialisation** | Random | Phase 1 checkpoint |

---

## Training — SD 1.5

### Phase 1

Edit `run_p1_sd15.sh` to set `CONDITION`, `DATASET_MODE`, and GPU, then:

```bash
bash run_p1_sd15.sh
```

Or run directly:

```bash
PYTHONPATH=. python scripts/train_p1_sd15.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="outputs/phase1/all" \
    --conditions="all" \
    --train_data_dir="data" \
    --dataset_mode="coco" \
    --mixed_precision="fp16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=50000 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --checkpointing_steps=5000 \
    --report_to="wandb" \
    --seed=42
```

### Phase 2

Edit `run_p2_sd15.sh`:

```bash
export ADAPTER_MODEL="outputs/phase1/all/checkpoint-50000/adapter"  # your Phase 1 checkpoint
export DATA_DIR="/path/to/your/dataset"
```

Then:

```bash
bash run_p2_sd15.sh
```

Or for a single condition:

```bash
PYTHONPATH=. python scripts/train_p2_sd15.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --pretrained_adapter_model_name_or_path="outputs/phase1/all/checkpoint-50000/adapter" \
    --output_dir="outputs/phase2/canny" \
    --conditions="canny" \
    --train_data_dir="data" \
    --dataset_mode="coco" \
    --mixed_precision="bf16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=50000 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --checkpointing_steps=5000 \
    --report_to="wandb" \
    --seed=42
```

> For segmentation fine-tuning always use `--dataset_mode=coco`.

---

## Training — SDXL

Identical structure but using the SDXL scripts:

```bash
# Phase 1
bash run_p1_sdxl.sh

# Phase 2
bash run_p2_sdxl.sh
```

Direct Phase 1 invocation:

```bash
PYTHONPATH=. python scripts/train_p1_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir="outputs/phase1/sdxl_all" \
    --conditions="all" \
    --train_data_dir="data" \
    --dataset_mode="coco" \
    --mixed_precision="fp16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=50000 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --checkpointing_steps=5000 \
    --seed=42
```

---

## Multi-GPU Training

```bash
# Configure accelerate once
accelerate config

# Launch (replace script name as needed)
accelerate launch scripts/train_p1_sd15.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="outputs/phase1/all" \
    --conditions="all" \
    --train_data_dir="data" \
    --dataset_mode="both" \
    --mixed_precision="fp16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=50000 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --seed=42
```

---

## Resuming Training

```bash
# Automatically pick the latest checkpoint
PYTHONPATH=. python scripts/train_p1_sd15.py \
    ... \
    --resume_from_checkpoint="latest"
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--pretrained_model_name_or_path` | required | HF model ID or local path |
| `--pretrained_adapter_model_name_or_path` | `None` | Phase 1 adapter checkpoint (Phase 2 only) |
| `--conditions` | `all` | `all` \| `canny` \| `depth_leres` \| `scribble_pidinet` \| `segmentation` |
| `--dataset_mode` | `coco` | `coco` \| `multigen` \| `both` |
| `--train_data_dir` | required | Dataset root directory |
| `--resolution` | `512` | Training image resolution |
| `--mixed_precision` | `no` | `fp16` or `bf16` |
| `--learning_rate` | `1e-5` | AdamW learning rate |
| `--max_train_steps` | `None` | Total optimiser steps |
| `--train_batch_size` | `2` | Per-GPU batch size |
| `--gradient_accumulation_steps` | `4` | Steps before one update |
| `--checkpointing_steps` | `5000` | Save checkpoint every N steps |
| `--resume_from_checkpoint` | `None` | Path or `"latest"` |
| `--gradient_checkpointing` | flag | Enable to save VRAM |
| `--report_to` | `wandb` | `wandb` \| `tensorboard` \| `none` |

---

## Inference

### SD 1.5

```python
import torch
from PIL import Image
from models.adapter import UNITY
from models.pipeline import StableDiffusionAdapterPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda"
model_id = "runwayml/stable-diffusion-v1-5"

vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
scheduler    = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

adapter = UNITY.from_pretrained("outputs/phase2/canny/adapter").to(device)

pipe = StableDiffusionAdapterPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
    unet=unet, adapter=adapter, scheduler=scheduler,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False,
)

result = pipe(
    prompt="a photo of a cat sitting on a sofa",
    image=Image.open("canny_map.png").convert("RGB"),
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]
result.save("output.png")
```

### SDXL

```python
import torch
from PIL import Image
from models.adapter import UNITY
from models.pipeline_sdxl import StableDiffusionXLAdapterPipeline

device   = "cuda"
adapter  = UNITY.from_pretrained("outputs/phase2/sdxl_canny/adapter").to(device, dtype=torch.float16)
pipe     = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    adapter=adapter, torch_dtype=torch.float16,
).to(device)

result = pipe(
    prompt="a futuristic city at night",
    image=Image.open("canny_map.png").convert("RGB"),
    num_inference_steps=50, guidance_scale=7.5,
).images[0]
result.save("output_sdxl.png")
```

---

## Memory Tips

| Technique | Flag |
|---|---|
| Mixed precision | `--mixed_precision="fp16"` or `"bf16"` |
| Gradient checkpointing | `--gradient_checkpointing` |
| Smaller batch | `--train_batch_size=1` |
| More grad accumulation | `--gradient_accumulation_steps=8` |
| xFormers attention | `pip install xformers` then it is used automatically |

---

## Citation

If you find UNITY useful in your research, please cite our paper:

```bibtex
@inproceedings{unity2026eccv,
  title     = {UNITY: Attention Flow Networks for Adaptive Conditioning in Diffusion},
  author    = {Das, Aryan and Biswas, Koushik and Abdar, Moloud and Verma, Vinay Kumar},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

---

## License

This project is released under the [Apache 2.0 License](LICENSE).
