export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
#export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"


accelerate launch train_fmri_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=500 --checkpointing_steps=5000 \
  --learning_rate=5.3e-05 --lr_scheduler="constant" --lr_warmup_steps=1000 \
  --seed=2022 \
  --output_dir="sd-pokemon-model-lora" \
  --report_to="wandb" \
  --filename="17"

# --mixed_precision="fp16"