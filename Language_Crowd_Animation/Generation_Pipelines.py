import os
import cv2
import time
import math
import copy
import shutil
import random
import logging
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import accelerate
import datasets
from dataclasses import dataclass
from packaging import version
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler, get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

class SgDistr_Generation_Pipeline():
    def __init__(self, text_encoder, tokenizer, scheduler, unet, config):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.unet = unet
        self.config = config

    @staticmethod
    def add_background_class(init_sem_map):
        new_added_dim = np.zeros((init_sem_map.shape[0], init_sem_map.shape[1], 1))
        bg_idx = np.argwhere(np.sum(np.array(init_sem_map), axis=2)==0)
        new_added_dim[bg_idx[:, 0], bg_idx[:, 1]] = np.array([1])
        return np.concatenate((np.array(init_sem_map), new_added_dim), axis=2)

    def inference(self, smaps, prompts, num_inference_steps, guidance_scale, save_path=None, show=False):
        assert self.scheduler.config.prediction_type == "epsilon"
        if save_path is not None and not os.path.exists(save_path):
            os.mkdir(save_path)
        torch_device = "cuda"
        weight_dtype = torch.float32
        if self.config.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.config.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.text_encoder.to(torch_device, dtype=weight_dtype)
        self.unet.to(torch_device, dtype=weight_dtype)

        prompts_ = prompts

        smaps_full = []
        for smp_id in range(len(smaps)):
            smaps_full.append(self.add_background_class(np.array(smaps[smp_id])))
        smaps_ = torch.from_numpy(np.array(smaps_full)).to(torch_device, dtype=weight_dtype)
        smaps_ = torch.permute(smaps_, (0, 3, 1, 2))

        generator = torch.Generator().manual_seed(self.config.seed)
        batch_size_ = len(prompts_)

        text_input = self.tokenizer(prompts_, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * batch_size_, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn((batch_size_, self.config.sg_distr_channels, self.config.map_size, self.config.map_size),generator=generator)
        latents = latents.to(torch_device, dtype=weight_dtype)
        latents = latents * self.scheduler.init_noise_sigma

        from tqdm.auto import tqdm
        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat((latents, smaps_), dim=1)
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latent_model_input] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        sgdistrs_all = []
        for dt_id in range(batch_size_):
            smap_i = np.array(torch.permute(smaps_[dt_id].clone().detach().cpu(), (1, 2, 0)))
            sg_distr_i = np.array(torch.permute(latents[dt_id].clone().detach().cpu(), (1, 2, 0)))
            sgdistrs_all.append(copy.deepcopy(sg_distr_i))
            text_i = prompts_[dt_id]

            from Utils import cv_visual_map
            sg_colors = np.array([[0,255,0],[0,0,255]])
            map_colors = np.concatenate([np.random.randint(0, 255, (len(smap_i[0][0])-2, 3)), sg_colors], axis=0)
            smap_cvimg = cv_visual_map(smap_i, colors=map_colors, save_nm=None, show=False)
            distr_cvimg = cv_visual_map(sg_distr_i, colors=sg_colors, save_nm=None, show=False)
            interval = np.ones((len(smap_cvimg), 5, 3))*255
            cat_img = cv2.hconcat([interval, smap_cvimg, interval, distr_cvimg, interval])
            cat_img_resize = cv2.resize(cat_img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            if show:
                print(text_i)
                cv2.imshow("img", cat_img_resize/255)
                cv2.waitKey(0)
            if save_path is not None:
                cv2.imwrite(os.path.join(save_path, "dt_"+str(dt_id)+".jpg"), cat_img_resize)
        if save_path is not None:
            np.save(os.path.join(save_path, "texts.npy"), {"texts": prompts_})

        return np.array(sgdistrs_all)


class Field_Generation_Pipeline():
    def __init__(self, text_encoder, tokenizer, scheduler, unet, config):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.unet = unet
        self.config = config

    @staticmethod
    def add_background_class(init_sem_map):
        new_added_dim = np.zeros((init_sem_map.shape[0], init_sem_map.shape[1], 1))
        bg_idx = np.argwhere(np.sum(np.array(init_sem_map), axis=2)==0)
        new_added_dim[bg_idx[:, 0], bg_idx[:, 1]] = np.array([1])
        return np.concatenate((np.array(init_sem_map), new_added_dim), axis=2)

    def inference(self, smaps, prompts, sg_distrs, num_inference_steps, guidance_scale, save_path=None, show=False):
        assert self.scheduler.config.prediction_type == "epsilon"
        if save_path is not None and not os.path.exists(save_path):
            os.mkdir(save_path)
        torch_device = "cuda"
        weight_dtype = torch.float32
        if self.config.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.config.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.text_encoder.to(torch_device, dtype=weight_dtype)
        self.unet.to(torch_device, dtype=weight_dtype)

        prompts_ = prompts

        smaps_full = []
        for smp_id in range(len(smaps)):
            smaps_full.append(self.add_background_class(np.array(smaps[smp_id])))
        smaps_ = torch.from_numpy(np.array(smaps_full)).to(torch_device, dtype=weight_dtype)
        smaps_ = torch.permute(smaps_, (0, 3, 1, 2))
        sg_distrs_ = torch.from_numpy(np.array(sg_distrs)).to(torch_device, dtype=weight_dtype)
        sg_distrs_ = torch.permute(sg_distrs_, (0, 3, 1, 2))

        generator = torch.Generator().manual_seed(self.config.seed)
        batch_size_ = len(prompts_)

        text_input = self.tokenizer(prompts_, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * batch_size_, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn((batch_size_, self.config.field_channels, self.config.map_size, self.config.map_size),generator=generator)
        latents = latents.to(torch_device, dtype=weight_dtype)
        latents = latents * self.scheduler.init_noise_sigma

        from tqdm.auto import tqdm
        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat((latents, smaps_, sg_distrs_), dim=1)
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latent_model_input] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        fields_all = []
        for dt_id in range(batch_size_):
            smap_i = np.array(torch.permute(smaps_[dt_id].clone().detach().cpu(), (1, 2, 0)))
            sg_distr_i = np.array(torch.permute(sg_distrs_[dt_id].clone().detach().cpu(), (1, 2, 0)))
            field_i = np.array(torch.permute(latents[dt_id].clone().detach().cpu(), (1, 2, 0)))
            fields_all.append(copy.deepcopy(field_i))
            text_i = prompts_[dt_id]

            from Utils import cv_visual_map, cv_visual_field
            sg_colors = np.array([[0,255,0],[0,0,255]])
            bg_color = np.array([[0,0,0]])
            map_colors = np.concatenate([np.random.randint(0, 255, (len(smap_i[0][0])-3, 3)), sg_colors, bg_color], axis=0)

            smap_cvimg = cv_visual_map(smap_i, colors=map_colors, save_nm=None, show=False)
            distr_cvimg = cv_visual_map(sg_distr_i, colors=sg_colors, save_nm=None, show=False)
            field_cvimg = cv_visual_field(field_i, grid_width=10, save_nm=None, show=False)

            enlarge_scale = int(len(field_cvimg)/len(smap_cvimg))
            smap_large = cv2.resize(smap_cvimg, None, fx=enlarge_scale, fy=enlarge_scale, interpolation=cv2.INTER_CUBIC)
            distr_large = cv2.resize(distr_cvimg, None, fx=enlarge_scale, fy=enlarge_scale, interpolation=cv2.INTER_CUBIC)

            interval = np.ones((len(smap_large), 20, 3))*255
            cat_img = cv2.hconcat([interval, smap_large, interval, distr_large, interval, field_cvimg, interval])

            if show:
                print(text_i)
                cv2.imshow("img", cat_img/255)
                cv2.waitKey(0)
            if save_path is not None:
                cv2.imwrite(os.path.join(save_path, "dt_"+str(dt_id)+".jpg"), cat_img)
        if save_path is not None:
            np.save(os.path.join(save_path, "texts.npy"), {"texts": prompts_})

        return np.array(fields_all)