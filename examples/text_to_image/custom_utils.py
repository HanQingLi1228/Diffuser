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
from pipeline_stable_diffusion_fmri import fMRI_StableDiffusionPipeline
from einops import rearrange, repeat
from torch.utils.tensorboard import SummaryWriter

def tensor_to_PIL(tensor, transform_train):
    unloader = transforms.ToPILImage()
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