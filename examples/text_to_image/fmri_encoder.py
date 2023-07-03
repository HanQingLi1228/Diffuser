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