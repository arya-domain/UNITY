clear
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export CONDITION="all" # canny # depth_leres # scribble_pidinet # segmentation
export OUTPUT_DIR="outputs/phase1/$CONDITION"
export DATASET_MODE="coco" # "coco", "multigen", or "both"
mkdir -p $OUTPUT_DIR

PYTHONPATH=. python scripts/train_p1_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --conditions=$CONDITION \
    --train_data_dir="data" \
    --dataset_mode=$DATASET_MODE \
    --mixed_precision="fp16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=50000 \
    --validation_steps=1000 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --checkpointing_steps=5000 \
    --report_to="wandb" \
    --seed=42
