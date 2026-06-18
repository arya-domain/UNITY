"""
Argument parser shared by all UNITY training scripts.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the UNITY universal condition adapter."
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path or HuggingFace model ID of the base diffusion model "
             "(e.g. 'runwayml/stable-diffusion-v1-5').",
    )
    parser.add_argument(
        "--pretrained_adapter_model_name_or_path",
        type=str,
        default=None,
        help="(Phase 2 only) Path to a Phase 1 adapter checkpoint directory "
             "or HuggingFace model ID to initialise from.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Optional path to a fine-tuned VAE to use instead of the one "
             "bundled with the base model.  When supplied the VAE is cast to "
             "the training mixed-precision dtype; otherwise it stays in fp32.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision / branch to load from HuggingFace Hub.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model weight variant (e.g. 'fp16') when loading from Hub.",
    )
    parser.add_argument(
        "--control_type",
        type=str,
        default="condition",
        help="Internal tag used to select adapter loading logic.  "
             "Leave as 'condition' for standard spatial control types.",
    )

    # ── Data ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Root directory that contains the 'coco/' and/or 'multigen20m/' "
             "sub-folders.",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="coco",
        choices=["coco", "multigen", "both"],
        help="Which dataset(s) to use for training.  "
             "'coco' = MS-COCO only;  "
             "'multigen' = MultiGen-20M only (no segmentation);  "
             "'both' = union of both.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="all",
        choices=["all", "canny", "depth_leres", "scribble_pidinet", "segmentation"],
        help="Condition type(s) to use for adapter conditioning.  "
             "Use 'all' for Phase 1 multi-condition pre-training; "
             "use a single condition key for Phase 2 fine-tuning.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Images (and condition maps) are resized to this square resolution "
             "before training.",
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where checkpoints and the final adapter weights are saved.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Sub-directory of output_dir used for experiment tracker logs.",
    )

    # ── Training ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Per-GPU training batch size.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of full passes over the training set.  Ignored when "
             "--max_train_steps is set.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of optimiser update steps.  Overrides "
             "--num_train_epochs when set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of forward passes to accumulate before one optimiser step.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to trade compute for lower VRAM usage.",
    )

    # ── Mixed precision ────────────────────────────────────────────────────────
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed-precision training mode.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 on Ampere GPUs for faster matmuls (slight precision loss).",
    )

    # ── Optimiser ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale --learning_rate by (batch_size × grad_accum × num_gpus).",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="Set gradients to None instead of zero after each optimiser step "
             "(saves a small amount of memory).",
    )

    # ── LR scheduler ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear", "cosine", "cosine_with_restarts",
            "polynomial", "constant", "constant_with_warmup",
        ],
        help="Learning-rate schedule type.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps at the start of training.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard restarts for cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor for polynomial LR decay.",
    )

    # ── Checkpointing ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help="Save an accelerate checkpoint every N optimiser steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep; oldest are deleted first.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to an accelerate checkpoint directory to resume from, "
             "or 'latest' to auto-select the most recent checkpoint in --output_dir.",
    )

    # ── Logging / validation ───────────────────────────────────────────────────
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["none", "tensorboard", "wandb", "all"],
        help="Experiment tracking backend.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help="Run a validation pass every N steps (if validation data is configured).",
    )

    return parser.parse_args()
import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_adapter_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained adapter model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--adapter_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained adapter model or model identifier from huggingface.co/models."
        " If not specified adapter weights are initialized w.r.t the configurations of SDXL.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="t2iadapter-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--detection_resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10e20,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading."),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--conditions", type=str, default="all", help="The conditions to use for training."
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the t2iadapter conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_t2iadapter",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    # if args.dataset_name is not None and args.train_data_dir is not None:
    #     raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the t2iadapter encoder."
        )

    return args


import math
from typing import List, Union, Optional, Dict, Tuple
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

ImageLike = Union[Image.Image, torch.Tensor, np.ndarray]

def _as_numpy_image(img: ImageLike, normalize: bool = True, resize: Optional[Tuple[int,int]] = None) -> np.ndarray:
    """
    Convert PIL / torch / numpy image to H x W x C numpy image (float32, range 0..1 for color images).
    - Accepts torch tensors of shape (C,H,W), (H,W), (B,C,H,W) (takes first), or numpy arrays similar shapes.
    - For single-channel images returns HxW (grayscale) or HxWx1 then converted to HxWx3 depending on usage.
    """
    # PIL.Image -> numpy
    if isinstance(img, Image.Image):
        arr = np.array(img)
    elif isinstance(img, torch.Tensor):
        if img.ndim == 4:  # (B,C,H,W) -> take first
            img = img[0]
        arr = img.detach().cpu().numpy()
    else:
        arr = np.array(img)

    # Channel-first conversion (C,H,W) -> (H,W,C)
    if arr.ndim == 3 and arr.shape[0] in (1,3,4) and arr.shape[2] not in (1,3,4):
        # Likely (C,H,W)
        arr = np.transpose(arr, (1,2,0))

    # If single-channel (H,W) -> keep as 2D (matplotlib handles it with cmap='gray')
    if arr.ndim == 2:
        # convert to float 0..1
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        else:
            arr = (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)
        if resize is not None:
            arr = np.array(Image.fromarray((arr*255).astype(np.uint8)).resize(resize)) / 255.0
        return arr

    # If has alpha channel, drop it
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    # Now arr should be (H,W,3)
    # Convert dtype/range to float32 0..1 for matplotlib
    if np.issubdtype(arr.dtype, np.floating):
        minv = float(np.nanmin(arr))
        maxv = float(np.nanmax(arr))
        if normalize:
            # common patterns: [-1,1], [0,1], [0,255]
            if minv >= -1.1 and maxv <= 1.1:
                # map min..max -> 0..1
                if abs(maxv - minv) < 1e-8:
                    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
                else:
                    arr = (arr - minv) / (maxv - minv)
            elif minv >= 0.0 and maxv <= 1.1:
                arr = np.clip(arr, 0.0, 1.0)
            else:
                # assume 0..255 floats
                arr = np.clip(arr / 255.0, 0.0, 1.0)
        else:
            arr = np.clip(arr, 0.0, 1.0)
    else:
        # integer types -> uint8 -> convert to 0..1 float
        arr = np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)

    if resize is not None:
        # PIL resize preserves channels correctly
        arr = np.array(Image.fromarray((arr*255).astype(np.uint8)).resize(resize)) / 255.0

    return arr.astype(np.float32)

def plot_image_grid(
    imgs: Union[List[ImageLike], Dict[str, ImageLike]],
    ncols: Optional[int] = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int,int] = (8,8),
    normalize: bool = True,
    resize: Optional[Tuple[int,int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    cmap: Optional[str] = None,
):
    """
    Plot images in a grid using matplotlib. Returns (fig, axes).

    Args:
        imgs: list of images or dict mapping title->image. Images may be PIL, torch.Tensor, or numpy.ndarray.
        ncols: number of columns. If None, a square-ish layout is used.
        titles: optional list of titles (ignored if imgs is dict).
        figsize: matplotlib figure size.
        normalize: whether to normalize float arrays (handles ranges like [-1,1], [0,1], [0,255]).
        resize: if provided, resize each image to (W,H) before plotting.
        save_path: if provided, saves the figure to that path.
        show: whether to call plt.show().
        cmap: optional colormap to force (leave None to let function auto-detect grayscale/color).
    """
    # If dict, extract items and use keys as titles
    if isinstance(imgs, dict):
        keys = list(imgs.keys())
        img_list = [imgs[k] for k in keys]
        titles = keys
    else:
        img_list = list(imgs)

    n = len(img_list)
    if n == 0:
        raise ValueError("No images to plot")

    if titles is not None and len(titles) != n:
        raise ValueError("titles length must match number of images")

    if ncols is None:
        ncols = int(math.ceil(math.sqrt(n)))
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    # Convert all to numpy images
    np_imgs = []
    is_gray = []
    for im in img_list:
        arr = _as_numpy_image(im, normalize=normalize, resize=resize)
        np_imgs.append(arr)
        # decide if grayscale (2D or 3D with last dim == 1 or R==G==B)
        if arr.ndim == 2:
            is_gray.append(True)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            is_gray.append(True)
            np_imgs[-1] = np_imgs[-1].squeeze(-1)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # check if constant across channels (R==G==B)
            if np.allclose(arr[...,0], arr[...,1]) and np.allclose(arr[...,1], arr[...,2]):
                is_gray.append(True)
                np_imgs[-1] = np_imgs[-1][...,0]
            else:
                is_gray.append(False)
        else:
            is_gray.append(False)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    # axes can be 2D or 1D depending on grid; flatten for easy indexing
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for idx in range(nrows * ncols):
        ax = axes_flat[idx]
        ax.axis('off')
        if idx < n:
            arr = np_imgs[idx]
            if is_gray[idx]:
                ax.imshow(arr, cmap=cmap or 'gray', vmin=0.0, vmax=1.0)
            else:
                # color image, ensure shape HxWx3
                ax.imshow(arr, vmin=0.0, vmax=1.0)
            if titles is not None:
                ax.set_title(str(titles[idx]), fontsize=10)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes
