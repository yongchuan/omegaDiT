import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
from typing import List, Optional, Union
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from json_label_dataset import JsonLabelDataset
from preprocessing.encoders import load_invae, load_vavae
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from samplers import euler_maruyama_sampler
from PIL import Image
from modelscope import CLIPModel, CLIPProcessor, CLIPConfig

logger = get_logger(__name__)


def get_clip_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
):
    """Get CLIP text embeddings for prompts.
    
    Reference: img_label_dataset.py get_clip_prompt_embeds()
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    # Use pooled output of CLIPTextModel
    # prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(device=device)

    return prompt_embeds


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    x = x / 255.
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(latents, latents_scale=1., latents_bias=0.):
    # With invae, we directly get sampled latents, no need to chunk into mean/std
    z = (latents * latents_scale + latents_bias) 
    return z 

# @torch.no_grad()
# def update_ema(ema_model, model, decay=0.9999):
#     """
#     Step the EMA model towards the current model.
#     """
#     ema_params = OrderedDict(ema_model.named_parameters())
#     model_params = OrderedDict(model.named_parameters())
#
#     for name, param in model_params.items():
#         name = name.replace("module.", "")
#         # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
#         ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 16 == 0, "Image size must be divisible by 16 (for the invae encoder)."
    latent_size = args.resolution // 16  # invae uses 16x downsampling

    if args.enc_type != None:
        # Only main process triggers the heavy I/O / download once
        if accelerator.is_main_process:
            _ = load_encoders(args.enc_type, device, args.resolution)

        # Make all processes wait until rank 0 is done
        accelerator.wait_for_everyone()

        # Now everyone can safely load from cache (or do whatever)
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
    else:
        raise NotImplementedError()

    # Load custom invae model
    if accelerator.is_main_process:
        # _ = load_invae(args.vae_name, device=device)
        _ = load_vavae("tokenizer/configs/vavae_f16d32_vfdinov2.yaml", device=device)

    # Make all processes wait until rank 0 is done
    accelerator.wait_for_everyone()
    # vae = load_invae(args.vae_name, device=device)
    vae = load_vavae("tokenizer/configs/vavae_f16d32_vfdinov2.yaml", device=device)
    vae.model.requires_grad_(False)
    channels = 32  # invae uses 32 channels

    # Load CLIP model for text encoding (only when using JsonLabelDataset)
    clip_model = None
    clip_tokenizer = None
    if args.use_json_dataset:
        clip_model_id = args.clip_model_id
        if accelerator.is_main_process:
            logger.info(f"Loading CLIP model: {clip_model_id}")
        clip_config = CLIPConfig.from_pretrained(clip_model_id)
        clip_model = CLIPModel.from_pretrained(
            clip_model_id, torch_dtype=torch.float32, config=clip_config
        ).to(device)
        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_processor = CLIPProcessor.from_pretrained(
            clip_model_id, padding="max_length", max_length=77,
            return_tensors="pt", truncation=True
        )
        clip_tokenizer = clip_processor.tokenizer
        clip_text_encoder = clip_model.text_model
    
    # invae uses 0.3099 scaling factor
    #scaling_factor = 0.3099
    #latents_scale = torch.tensor([scaling_factor] * channels).view(1, channels, 1, 1).to(device)
    #latents_bias = torch.zeros(channels).view(1, channels, 1, 1).to(device)

    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=channels,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        **block_kwargs
    )

    model = model.to(device)
    #ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    #requires_grad(ema, False)
    

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        weighting=args.weighting,
        cfm_weighting=args.cfm_weighting,
        apply_time_shift=args.time_shifting,
        shift_base=args.shift_base,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    if args.use_json_dataset:
        train_dataset = JsonLabelDataset(args.data_dir, label_file=args.label_file)
        if accelerator.is_main_process:
            logger.info(f"Using JsonLabelDataset with label file: {args.label_file}")
    else:
        train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    # update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    # ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu', weights_only=False,
            )
        model.load_state_dict(ckpt['model'])
        # ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REG",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )

        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    #sample_batch_size = 64 // accelerator.num_processes
    #gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    #assert gt_raw_images.shape[-1] == args.resolution
    #gt_xs = gt_xs[:sample_batch_size]
    #gt_xs = sample_posterior(
        #gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
       # )
    #ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    #ys = ys.to(device)
    # Create sampling noise:
    #n = ys.size(0)
    #xT = torch.randn((n, channels, latent_size, latent_size), device=device)
    #cls_z = torch.randn((n, encoders[0].embed_dim), device=device)
    
    if accelerator.is_main_process:
        sample_dir = os.path.join(args.output_dir, args.exp_name, "samples")
        os.makedirs(sample_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            
            # Handle labels based on dataset type
            if args.use_json_dataset:
                # y is a list of text captions, convert to CLIP embeddings
                with torch.no_grad():
                    labels = get_clip_prompt_embeds(
                        clip_tokenizer, clip_text_encoder, y, device=device
                    )
            else:
                # y is already class labels tensor
                y = y.to(device)
                labels = y
            
            z = None
            with torch.no_grad():
                #x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'dinov2' in encoder_type:
                            dense_z = z['x_norm_patchtokens']
                            #cls_token = z['x_norm_clstoken']
                            #dense_z = torch.cat([cls_token.unsqueeze(1), dense_z], dim=1)
                        else:
                            exit()
                        zs.append(dense_z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss1, proj_loss1, time_input, noises, cfm_loss = loss_fn(model, x, model_kwargs, zs=zs,
                                                                       cls_token=None,
                                                                       time_input=None, noises=None)
                loss_mean = loss1.mean()
                proj_loss_mean = proj_loss1.mean() * args.proj_coeff
                cfm_loss_mean = cfm_loss.mean() * args.cfm_coeff
                loss = loss_mean + proj_loss_mean + cfm_loss_mean 


                ## optimization
                accelerator.backward(loss)
                #if accelerator.sync_gradients:
                    #params_to_clip = model.parameters()
                    #grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # if accelerator.sync_gradients:
                #     update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.state_dict(),
                        # "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            logs = {
                "loss_final": accelerator.gather(loss).mean().detach().item(),
                "loss_mean": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                #"loss_mean_cls": accelerator.gather(loss_mean_cls).mean().detach().item(),
                "cfm_loss": accelerator.gather(cfm_loss_mean).mean().detach().item()
                #"grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }

            log_message = ", ".join(f"{key}: {value:.6f}" for key, value in logs.items())
            logging.info(f"Step: {global_step}, Training Logs: {log_message}")

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=2000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ops-head", type=int, default=16)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=8)#256
    parser.add_argument("--vae-name", type=str, default="REPA-E/e2e-invae")
    parser.add_argument("--use-json-dataset", action="store_true", help="Use JsonLabelDataset with text captions")
    parser.add_argument("--label-file", type=str, default=None, help="Path to JSON label file for JsonLabelDataset")
    parser.add_argument("--clip-model-id", type=str, default="AI-ModelScope/CLIP-GmP-ViT-L-14", help="CLIP model ID for text encoding")

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--cls", type=float, default=0.03)
    parser.add_argument("--cfm-weighting", default="uniform", choices=["uniform", "linear"], type=str)
    parser.add_argument("--cfm-coeff", type=float, default=0.05)

    parser.add_argument("--time-shifting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shift-base", type=int, default=4096)

    # sampling specific
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale for in-training sampling.")
    parser.add_argument("--cls-cfg-scale", type=float, default=1.0, help="CLS guidance scale (used inside sampler).")
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--num-sample-steps", type=int, default=50, help="Diffusion sampling steps for in-training sampling.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)

