clear

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export ADAPTER_MODEL="RESULTS_X/all/checkpoint-10000/adapter"
export DATA_DIR="/efs/drsanny/visual_extension/storage/Projects/UniSpec/data"

run_experiment () {
    local dataset=$1
    local condition=$2
    local gpu_id=$3

    echo ">>> Starting $dataset - $condition on GPU $gpu_id"
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export DATASET_MODE=$dataset
    export CONDITION=$condition
    export OUTPUT_DIR="RESULTS_Y_NEW/${dataset}_${condition}"
    mkdir -p "$OUTPUT_DIR"

    # 
    python train_p2.py \
        --pretrained_model_name_or_path=$MODEL_DIR \
        --pretrained_adapter_model_name_or_path=$ADAPTER_MODEL \
        --output_dir=$OUTPUT_DIR \
        --conditions=$CONDITION \
        --train_data_dir=$DATA_DIR \
        --mixed_precision="bf16" \
        --resolution=512 \
        --learning_rate=1e-5 \
        --max_train_steps=50000 \
        --validation_steps=1000 \
        --train_batch_size=2 \
        --gradient_accumulation_steps=4 \
        --report_to="wandb" \
        --seed=42 &
}

# Run 4 jobs in parallel on GPUs 0,1,2,3
# run_experiment "both" "canny" 0
# run_experiment "both" "depth_leres" 1
# run_experiment "both" "scribble_pidinet" 2
run_experiment "coco" "segmentation" 3

# Wait for all to finish
wait

echo "All 4 experiments completed!"
