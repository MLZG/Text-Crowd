import sys
sys.path.append('../')

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

logger = get_logger(__name__, log_level="INFO")

@dataclass
class Training_Config:
    # configs of dataset
    data_path = "./Dataset/Data_Full_V1/"
    obj_nums = [0, 1, 2, 3, 4, 5]
    group_nums = [1, 2, 3]
    data_scale = 1.0

    # configs for training
    input_perturbation = 0    # The scale of input perturbation. Recommended 0.1.
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    revision = None
    output_dir = "SgDistr-Full-V1"
    logging_dir = "logs"
    tracker_project_name = "SgDistr"
    report_to = "tensorboard"
    seed = 0    # A seed for reproducible training.

    map_size = 64
    smap_channels = 9
    sg_distr_channels = 2

    dataloader_num_workers = 4
    train_batch_size = 8
    num_train_epochs = 50
    max_train_steps = None
    validation_epochs = 1
    save_pipeline_epochs = 5

    gradient_accumulation_steps = 8    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_checkpointing = True   # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
    mixed_precision = "no"  # "no", "fp16", "bf16"

    learning_rate = 3e-5
    scale_lr = False     # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    lr_scheduler = "constant"
    lr_warmup_steps = 500

    use_8bit_adam = False    # Whether or not to use 8-bit Adam from bitsandbytes.
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    max_grad_norm = 1.0

    checkpointing_steps = 1000 # Save a checkpoint of the training state. Can only used for resuming training
    checkpoints_total_limit = None # the max num of checkpoints (old checkpoints will be removed if the number of checkpoints is over this limit)
    resume_from_checkpoint = None # Use a path saved by "--checkpointing_steps", or "latest"

    enable_xformers_memory_efficient_attention = False   # Whether or not to use xformers.

    # The prediction_type that shall be used for training.
    # Choose between 'epsilon' or 'v_prediction' or leave `None`.
    # If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.
    prediction_type = None  # the prediction_type of noise scheduler in stable diffusion v1.5 is default to be "epsilon".

    nohup_tqdm_update_interval = 60 # 60 if use nohup; None if run with local terminal or tmux


def load_dataset(config):
    dataset_ = []
    mapping_ = []
    for obj_n in config.obj_nums:
        for group_n in config.group_nums:
            print("### reading data of obj %d, group %d..." %(obj_n, group_n))
            dt_path_og = config.data_path + "Obj" + str(obj_n) + "_" + "Group" + str(group_n) + "/" + "dt_training.npy"
            print("data path:", dt_path_og)
            dt_og = np.load(dt_path_og, allow_pickle=True).item()

            assert abs(int(len(dt_og["data"])*config.data_scale) - len(dt_og["data"])*config.data_scale) < 1e-6
            data_size_foruse = int(len(dt_og["data"])*config.data_scale)
            print("total data size: %d | data size for use: %d" %(len(dt_og["data"]), data_size_foruse))
            for dt_i in dt_og["data"][0:data_size_foruse]:
                dataset_.append({"semantic_map": dt_i["semantic_map"], "group_descriptions": dt_i["group_descriptions"],
                                 "group_sg_distrs": dt_i["group_sg_distrs"]})
                for g_id in range(dt_i["group_num"]):
                    mapping_.append((len(dataset_)-1, g_id))
            print("current dataset size: %d | current group (real dataset) size: %d" %(len(dataset_), len(mapping_)))
    return dataset_, mapping_


# Train a UNet2DConditionModel which generates the start_goal distributions with diffusion
# Unet input:
#   semantic map (64*64*9)  &  noisy sg_distr (64*64*2)
#   text embeddings (condition)
# Unet Output:
#   predicted noise for denosing the sg_distr (64*64*2)
def train(config):
    # set accelerator
    logging_dir = os.path.join(config.output_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
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
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer", revision=config.revision)

    def deepspeed_zero_init_disabled_context_manager():
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision)


    unet = UNet2DConditionModel(
        sample_size=config.map_size,
        in_channels=config.smap_channels+config.sg_distr_channels,
        out_channels=config.sg_distr_channels,
        block_out_channels=(80, 160, 320, 320),
        norm_num_groups=16,
        encoder_hid_dim=768,
    )

    # Freeze text_encoder and set unet to be trainable
    text_encoder.requires_grad_(False)
    unet.set_attention_slice("max")     # maximum amount of memory is saved by running only one slice at a time
    unet.train()

    # set xformers
    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn("xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
        def load_model_hook(models, input_dir):
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

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if config.scale_lr:
        config.learning_rate = (config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes)

    # Initialize the optimizer
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )


    # load dataset and pre-process the dataset, get dataloader.
    with accelerator.main_process_first():
        # get dataset and mapping
        dataset_, mapping_ = load_dataset(config)

        ### Pre-process the dataset
        # Note that sem_maps are NOT standard one-hot maps: the feature for background is all-zero vectors.
        # A pre-process is needed to convert them into standard one-hot maps.
        # sg_distrs are not needed for this process as it can be seen as the start and goal distribution separately.
        # We also need to tokenize input texts.
        def add_background_class(init_sem_map):
            new_added_dim = np.zeros((init_sem_map.shape[0], init_sem_map.shape[1], 1))
            bg_idx = np.argwhere(np.sum(np.array(init_sem_map), axis=2)==0)
            new_added_dim[bg_idx[:, 0], bg_idx[:, 1]] = np.array([1])
            return np.concatenate((np.array(init_sem_map), new_added_dim), axis=2)
        for dt_id in range(len(dataset_)):
            dataset_[dt_id]["semantic_map"] = add_background_class(dataset_[dt_id]["semantic_map"])
            dataset_[dt_id]["group_tokens"] = tokenizer(dataset_[dt_id]["group_descriptions"],
                                                        max_length=tokenizer.model_max_length,
                                                        padding="max_length",
                                                        truncation=True,
                                                        return_tensors="pt").input_ids
        class Dataset_Crowd(Dataset):
            def __init__(self, data_, mapping_):
                self.data = data_
                self.mapping = mapping_
            def __len__(self):
                return len(self.mapping)
            def __getitem__(self, index):
                (data_id, group_id) = self.mapping[index]
                return np.array(self.data[data_id]["semantic_map"]).astype(np.float32), self.data[data_id]["group_tokens"][group_id],\
                       self.data[data_id]["group_descriptions"][group_id], np.array(self.data[data_id]["group_sg_distrs"][group_id]).astype(np.float32)

        dataset_crowd = Dataset_Crowd(dataset_, mapping_)

    dataloader_ = DataLoader(dataset=dataset_crowd, batch_size=config.train_batch_size, shuffle=True,
                             num_workers=config.dataloader_num_workers, drop_last=False)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader_)/config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, dataloader_, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader_, lr_scheduler)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.mixed_precision = accelerator.mixed_precision

    # Move text_encode to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader_) / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict((nm,getattr(config, nm)) for nm in dir(config) if not nm.startswith("__"))
        tracker_config.pop("obj_nums")
        tracker_config.pop("group_nums")
        accelerator.init_trackers(config.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset_)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run.")
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    m_interval = 0.1 if config.nohup_tqdm_update_interval is None else config.nohup_tqdm_update_interval
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        mininterval=m_interval
    )

    for epoch in range(first_epoch, config.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(dataloader_):
            with accelerator.accumulate(unet):
                sem_maps, text_tokens, text_list_usls, sg_distrs = batch
                sem_maps = torch.permute(sem_maps, (0, 3, 1, 2))
                sg_distrs = torch.permute(sg_distrs, (0, 3, 1, 2))

                # Sample noise that we'll add to the sg_distrs
                noise = torch.randn_like(sg_distrs)
                if config.input_perturbation:
                    new_noise = noise + config.input_perturbation * torch.randn_like(noise)
                bsz = sg_distrs.shape[0]
                # Sample a random timestep for each sg_distrs
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=sg_distrs.device)
                timesteps = timesteps.long()

                # Add noise to the sg_distrs according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if config.input_perturbation:
                    noisy_sg_distrs = noise_scheduler.add_noise(sg_distrs, new_noise, timesteps)
                else:
                    noisy_sg_distrs = noise_scheduler.add_noise(sg_distrs, noise, timesteps)

                unet_input = torch.cat((noisy_sg_distrs, sem_maps),dim=1)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(text_tokens)[0]

                # Get the target for loss depending on the prediction type
                if config.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=config.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(sg_distrs, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample

                # Compute loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # (for logging) Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # save the checkpoint
                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # Before saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # Before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break
        # Training of one epoch ends here.

        # validation
        if accelerator.is_main_process:
            if epoch % config.validation_epochs == 0:
                logger.info("Running validation... ")
                vld_pipeline = SgDistr_Generation_Pipeline(text_encoder=text_encoder, tokenizer=tokenizer,
                                                           scheduler=noise_scheduler, unet=accelerator.unwrap_model(unet),
                                                           config=config)
                dt_validation = np.load(config.data_path+"Obj5_Group1/dt_validation.npy", allow_pickle=True).item()
                dt_single = dt_validation["data"][0]
                vld_pipeline.inference(smaps=[dt_single["semantic_map"]], prompts=[dt_single["group_descriptions"][0]],
                                       num_inference_steps=20, guidance_scale=1,
                                       save_path=os.path.join(config.output_dir, f"validation-{global_step}"), show=False)
                del vld_pipeline
                torch.cuda.empty_cache()

        # pipeline saving
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if epoch % config.save_pipeline_epochs == 0:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    config.pretrained_model_name_or_path,
                    text_encoder=text_encoder,
                    unet=accelerator.unwrap_model(unet),
                    revision=config.revision,
                    safety_checker=None
                )
                pipeline.safety_checker = None
                pipeline.requires_safety_checker = False
                saving_path = os.path.join(config.output_dir, f"pipeline-{epoch}-{global_step}")
                if not os.path.exists(saving_path):
                    os.mkdir(saving_path)
                pipeline.save_pretrained(saving_path)
                del pipeline

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            unet=unet,
            revision=config.revision,
            safety_checker=None
        )
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        saving_path = os.path.join(config.output_dir, "pipeline-final")
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
        pipeline.save_pretrained(saving_path)
    accelerator.end_training()


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

        for dt_id in range(batch_size_):
            smap_i = np.array(torch.permute(smaps_[dt_id].clone().detach().cpu(), (1, 2, 0)))
            sg_distr_i = np.array(torch.permute(latents[dt_id].clone().detach().cpu(), (1, 2, 0)))
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


def evaluation(config):
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path,
                                                 subfolder="text_encoder",
                                                 revision=config.revision,
                                                 use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path,
                                              subfolder="tokenizer",
                                              revision=config.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path,
                                                    subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(config.output_dir+"-PreTest1/checkpoint-37000",
                                                subfolder="unet",
                                                use_safetensors=True)

    pipeline_ = SgDistr_Generation_Pipeline(text_encoder=text_encoder, tokenizer=tokenizer,
                                               scheduler=noise_scheduler, unet=unet,
                                               config=config)
    dt_validation = np.load(config.data_path+"Obj5_Group1/dt_validation.npy", allow_pickle=True).item()
    dt_id = 13
    group_id = 0

    dt_single = dt_validation["data"][dt_id]
    pipeline_.inference(smaps=[dt_single["semantic_map"]], prompts=[dt_single["group_descriptions"][group_id]],
                           num_inference_steps=20, guidance_scale=1,
                           save_path=None, show=True)
    from Utils import cv_visual_map
    cv_visual_map(dt_single["group_sg_distrs"][group_id], show=True)




if __name__ == '__main__':
    training_config = Training_Config()
    train(training_config)
    # evaluation(training_config)
