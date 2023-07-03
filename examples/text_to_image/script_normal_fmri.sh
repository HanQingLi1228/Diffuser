export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
#export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export PATH="/usr/local/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"

#accelerate launch --mixed_precision="fp16"  fmri2img_infer.py \
accelerate launch --mixed_precision="fp16"  train_fmri_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=5 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=30000 \
  --learning_rate=5.3e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=100 \
  --use_8bit_adam \
  --filename="6"