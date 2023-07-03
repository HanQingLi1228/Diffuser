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
sys.path.append('/home/hanqingli/diffusers/examples/text_to_image')
from custom_utils import *
from fmri_encoder import *
from pipeline_stable_diffusion_fmri import fMRI_StableDiffusionPipeline
from einops import rearrange, repeat
from torch.utils.tensorboard import SummaryWriter


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}



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


def main():
    args = parse_args()
    filename = "/home/hanqingli/diffusers/experiment/normal_fmri/" + str(args.filename)
    mkdir(filename)
    writer = SummaryWriter(filename+"/loss")
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
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

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
     
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, low_cpu_mem_usage=False
    )
    # import pdb
    # pdb.set_trace()
    # 定义fMRI encoder
    pretrain_mbm_path = "/data/hanqingli/pretrains/GOD/fmri_encoder_just_model.pth"
    model_dict = torch.load(pretrain_mbm_path, map_location='cpu')
    fmri_encoder = fmri_cond_model(model_dict, num_voxels=4656, cond_dim=1024, global_pool=False, dtype=torch.float16)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    fmri_encoder.requires_grad_(True)
    unet.time_embed_condition.requires_grad_(True)

    # # Create EMA for the unet.
    # if args.use_ema:
    #     ema_unet = UNet2DConditionModel.from_pretrained(
    #         args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    #     )
    #     ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warn(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

    # def compute_snr(timesteps):
    #     """
    #     Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    #     """
    #     alphas_cumprod = noise_scheduler.alphas_cumprod
    #     sqrt_alphas_cumprod = alphas_cumprod**0.5
    #     sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    #     # Expand the tensors.
    #     # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    #     sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    #     while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
    #         sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    #     alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    #     sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    #     while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
    #         sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    #     sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    #     # Compute SNR.
    #     snr = (alpha / sigma) ** 2
    #     return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    parameters = list(unet.parameters()) + list(fmri_encoder.parameters())
    optimizer = optimizer_cls(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    #############

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
    train_dataloader = torch.utils.data.DataLoader(fmri_latents_dataset_train, batch_size=5, shuffle=True, num_workers=args.dataloader_num_workers)
    test_dataloader = torch.utils.data.DataLoader(fmri_latents_dataset_test, batch_size=len(fmri_latents_dataset_test), shuffle=False, num_workers=args.dataloader_num_workers)

    #############

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, fmri_encoder, test_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, fmri_encoder, test_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(fmri_latents_dataset_train)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    ssss = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        # torch.save(vae.state_dict(), "/data/hanqingli/diffusers/origin/vae.pth")
        # torch.save(unet.state_dict(), "/data/hanqingli/diffusers/origin/unet.pth")
        # torch.save(fmri_encoder.state_dict(), "/data/hanqingli/diffusers/origin/fmri_encoder.pth")
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            ssss += 1
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["image"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = fmri_encoder(batch["fmri"]).to(dtype=weight_dtype)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                writer.add_scalar("train_loss", train_loss, ssss)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # if global_step % args.checkpointing_steps == 0:
                #     if accelerator.is_main_process:
                #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #         accelerator.save_state(save_path)
                #         logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            writer.add_scalar("step_loss", loss, ssss)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break 
        # torch.save(vae.state_dict(), "/data/hanqingli/diffusers/after/vae.pth")
        # torch.save(unet.state_dict(), "/data/hanqingli/diffusers/after/unet.pth")
        # torch.save(fmri_encoder.state_dict(), "/data/hanqingli/diffusers/after/fmri_encoder.pth")
        # import pdb
        # pdb.set_trace() 
        if accelerator.is_main_process:
            if  epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                logger.info("Running validation... ")
                pipeline = fMRI_StableDiffusionPipeline.from_pretrained(
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
                if args.enable_xformers_memory_efficient_attention:
                    pipeline.enable_xformers_memory_efficient_attention()
                if args.seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                if epoch % 10 == 0:
                    print("run full validation")
                    all_images = []
                    for step, batch in enumerate(test_dataloader):
                        #batch_f = batch['fmri'].reshape(batch['fmri'].shape[0], batch['fmri'].shape[-1])
                        cc = 0
                        for gt, data in zip(batch['image'], batch['fmri']):
                            cc += 1
                            # if cc > 2:
                            #     break
                            images = []
                            gt = tensor_to_PIL(gt, img_transform_train)
                            images.append(gt)
                            data = repeat(data, 'h w -> b h w', b=1)
                            for _ in range(5):
                                images.append(
                                    pipeline(data, num_inference_steps=250, generator=generator, guidance_scale = 1).images[0]
                                )
                            save_img(images, direction="c", num=cc, epoch=epoch, filename=filename)
                            all_images.append(images)
                    save_img(all_images, direction="c", num=epoch, epoch=epoch, filename=filename)
                else:
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
                # for tracker in accelerator.trackers:
                #     if tracker.name == "tensorboard":
                #         np_images = np.stack([np.asarray(img) for img in images])
                #         tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                #     elif tracker.name == "wandb":
                #         tracker.log(
                #             {
                #                 "validation": [
                #                     wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                #                     for i, image in enumerate(images)
                #                 ]
                #             }
                #         )
                #     else:
                #         logger.warn(f"image logging not implemented for {tracker.name}")

                del pipeline
                torch.cuda.empty_cache()

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())
        # torch.save(vae.state_dict(), "/data/hanqingli/diffusers/after/vae.pth")
        # torch.save(unet.state_dict(), "/data/hanqingli/diffusers/after/unet.pth")
        # torch.save(fmri_encoder.state_dict(), "/data/hanqingli/diffusers/after/fmri_encoder.pth")
        # import pdb
        # pdb.set_trace()

    # Create the pipeline using the trained modules and save it.
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = accelerator.unwrap_model(unet)
    #     if args.use_ema:
    #         ema_unet.copy_to(unet.parameters())

    #     pipeline = fMRI_StableDiffusionPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         text_encoder=fmri_encoder,
    #         vae=vae,
    #         unet=unet,
    #         revision=args.revision,
    #     )
    #     pipeline.save_pretrained(args.output_dir)

    #     if args.push_to_hub:
    #         upload_folder(
    #             repo_id=repo_id,
    #             folder_path=args.output_dir,
    #             commit_message="End of training",
    #             ignore_patterns=["step_*", "epoch_*"],
    #         )

    accelerator.end_training()


if __name__ == "__main__":
    main()
