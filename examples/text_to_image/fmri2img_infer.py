#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from PIL import Image
if is_wandb_available():
    import wandb

import sys
sys.path.append('/home/hanqingli/diffusers/src/diffusers/dataset')
from fmri_dataset import *
import copy
sys.path.append('/home/hanqingli/diffusers/src/diffusers/models')
from fMRI_mae import fmri_encoder
sys.path.append('/home/hanqingli/diffusers/src/diffusers/pipelines/stable_diffusion')
from pipeline_stable_diffusion_fmri import fMRI_StableDiffusionPipeline
from einops import rearrange, repeat


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0.dev0")

logger = get_logger(__name__, log_level="INFO")

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor, transform_train):
    #tensor = F.interpolate(tensor, scale_factor=2, mode='nearest')
    tensor = (tensor + 1) / 2 
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    
    return image

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")


def save_img(ims, direction, num, filename, epoch):
    image_path = filename+"/image/epoch_{}".format(epoch)
    mkdir(image_path)
 
    if type(ims[0]) is list:
        final = []
        for images in ims:
            width, height = images[0].size
            result = Image.new(images[0].mode, (width * len(images), height))
            for i, im in enumerate(images):
                result.paste(im, box=(i * width, 0))
            final.append(result)
        final_result = Image.new(ims[0][0].mode, (width * len(ims[0]), height * len(ims)))
        for i, im in enumerate(final):
            final_result.paste(im, box=(0, i * height))
 
        # 保存图片
        final_result.save(image_path+"/result_{}_epoch.jpg".format(epoch))
    else:
        width, height = ims[0].size
        result = Image.new(ims[0].mode, (width * len(ims), height))
 
        # 拼接图片
        for i, im in enumerate(ims):
            result.paste(im, box=(i * width, 0))
 
        # 保存图片
        result.save(image_path+"/test{}.jpg".format(num))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
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
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
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
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
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
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        default=1e-4,
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
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--filename", type=str, default="no_filename")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def create_fmri_cond_model(num_voxels, global_pool, config=None):
    model = fmri_encoder(num_voxels=num_voxels, patch_size=16, embed_dim=1024,
                depth=24, num_heads=16, mlp_ratio=1.0, global_pool=global_pool) 
    return model

class fmri_cond_model(nn.Module):
    def __init__(self, model_dict, num_voxels, cond_dim=1280, global_pool=True, dtype=torch.float32):
        super().__init__()
        # prepare pretrained fmri mae 
        model = create_fmri_cond_model(num_voxels, global_pool=global_pool, config=None)
        model.load_checkpoint(model_dict['model'])
        self.mae = model
        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool
        self.dtype = dtype

    def forward(self, x):
        # n, c, w = x.shape
        # print(self.mae.blocks[0].attn.qkv.weight)
        latent_crossattn = self.mae(x)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out


def main():
    args = parse_args()
    filename = "/home/hanqingli/diffusers/experiment/normal_fmri/" + str(args.filename)
    mkdir(filename)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    import pdb
    pdb.set_trace()
    unet = UNet2DConditionModel.from_pretrained(
        "/home/hanqingli/mind-vis-ldm/finetuned", revision=args.non_ema_revision, low_cpu_mem_usage=False
    )
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
     
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, low_cpu_mem_usage=False
    )

    # 定义fMRI encoder
    pretrain_mbm_path = "/data/hanqingli/pretrains/GOD/fmri_encoder_just_model.pth"
    model_dict = torch.load(pretrain_mbm_path, map_location='cpu')
    fmri_encoder = fmri_cond_model(model_dict, num_voxels=4656, cond_dim=1024, global_pool=False, dtype=torch.float16)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
        x_aug = copy.deepcopy(x)
        idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
        x_aug[idx] = 0
        return torch.FloatTensor(x_aug)

    def normalize(img):
        if img.shape[-1] == 3:
            img = rearrange(img, 'h w c -> c h w')
        img = torch.tensor(img)
        img = img * 2.0 - 1.0 # to -1 ~ 1
        return img
    
    class random_crop:
        def __init__(self, size, p):
            self.size = size
            self.p = p
        def __call__(self, img):
            if torch.rand(1) < self.p:
                return transforms.RandomCrop(size=(self.size, self.size))(img)
            return img

    def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')
    # diffuser中不存在crop， 后期可以去掉看看效果
    crop_ratio = 0.2
    img_size = 256
    crop_pix = int(crop_ratio*img_size)
    img_transform_train = transforms.Compose([
        normalize,
        random_crop(img_size-crop_pix, p=0.5),  
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    ])
    img_transform_test = transforms.Compose([
        normalize, 
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    ])

    fmri_latents_dataset_train, fmri_latents_dataset_test = create_Kamitani_dataset('/home/hanqingli/diffusers/data/Kamitani/npz', 'VC', 16, 
            fmri_transform=fmri_transform, image_transform=[img_transform_train, img_transform_test], 
            subjects=['sbj_3'])
    num_voxels = fmri_latents_dataset_train.num_voxels
    test_dataloader = torch.utils.data.DataLoader(fmri_latents_dataset_test, batch_size=len(fmri_latents_dataset_test), shuffle=False, num_workers=args.dataloader_num_workers)

    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=fmri_encoder,
        unet=accelerator.unwrap_model(unet),
        revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    # if epoch % 10 == 0:
    #     print("run full validation")
    #     all_images = []
    #     for step, batch in enumerate(test_dataloader):
    #         #batch_f = batch['fmri'].reshape(batch['fmri'].shape[0], batch['fmri'].shape[-1])
    #         cc = 0
    #         for gt, data in zip(batch['image'], batch['fmri']):
    #             cc += 1
    #             # if cc > 2:
    #             #     break
    #             images = []
    #             gt = tensor_to_PIL(gt, img_transform_train)
    #             images.append(gt)
    #             data = repeat(data, 'h w -> b h w', b=1)
    #             for _ in range(5):
    #                 images.append(
    #                     pipeline(data, num_inference_steps=250, generator=generator, guidance_scale = 1).images[0]
    #                 )
    #             save_img(images, direction="c", num=cc, epoch=epoch, filename=filename)
    #             all_images.append(images)
    #     save_img(all_images, direction="c", num=epoch, epoch=epoch, filename=filename)

    print("run sample validation")
    all_images = []
    for step, batch in enumerate(test_dataloader):
        #batch_f = batch['fmri'].reshape(batch['fmri'].shape[0], batch['fmri'].shape[-1])
        cc = 0
        for gt, data in zip(batch['image'], batch['fmri']):
            cc += 1
            # if cc > 2:
            #     break
            if cc not in [15, 16, 17]:
                continue
            images = []
            gt = tensor_to_PIL(gt, img_transform_train)
            images.append(gt)
            data = repeat(data, 'h w -> b h w', b=1)
            for _ in range(1):
                images.append(
                    pipeline(data, num_inference_steps=250, generator=generator, guidance_scale = 1).images[0]
                )
            save_img(images, direction="c", num=cc, epoch=epoch, filename=filename)
            all_images.append(images)
    save_img(all_images, direction="c", num=epoch, epoch=epoch, filename=filename)

if __name__ == "__main__":
    main()
