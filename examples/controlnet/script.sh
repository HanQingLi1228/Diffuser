export MODEL_DIR="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="result"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --tracker_project_name="controlnet-demo" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4