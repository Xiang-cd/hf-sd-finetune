export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

ADDR=${1:-127.0.0.1}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
MASTER_PORT=${MASTER_PORT:-29506}

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
echo $ADDR
accelerate_args="--num_machines $WORLD_SIZE \
                 --machine_rank $RANK --num_processes 1 \
                 --main_process_port $MASTER_PORT \
                 --main_process_ip $ADDR"


accelerate launch --mixed_precision="fp16" $accelerate_args  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=15000 \
  --checkpointing_steps=2000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model"
#   --gradient_checkpointing \