import sys
from dataclasses import dataclass
from typing import Union, List
import os
import time
import torch
from rich.console import Console
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
import PIL.Image
import ipdb
import cv2
CONSOLE = Console(width=180)
CONST_SCALE = 0.18215
finetune_module_dir = os.path.dirname(os.path.realpath(__file__))
diffusion_cache_dir = os.path.join(finetune_module_dir, 'cache_dir')

from diffusers import (
    DDIMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionInstructPix2PixPipeline,
)
from transformers import logging


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import PIL_INTERPOLATION


from torchvision.transforms import Resize

def preprocess_mask(mask, batch_size, scale_factor=8):
    valid_mask_channel_sizes = [1, 3]
    # if mask channel is fourth tensor dimension, permute dimensions to pytorch standard (B, C, H, W)
    if mask.shape[3] in valid_mask_channel_sizes:
        mask = mask.permute(0, 3, 1, 2)
    elif mask.shape[1] not in valid_mask_channel_sizes:
        raise ValueError(
            f"Mask channel dimension of size in {valid_mask_channel_sizes} should be second or fourth dimension,"
            f" but received mask of shape {tuple(mask.shape)}"
        )
    # (potentially) reduce mask channel dimension from 3 to 1 for broadcasting to latent shape
    mask = mask.mean(dim=1, keepdim=True)
    h, w = mask.shape[-2:]
    h, w = (x - x % 8 for x in (h, w))  # resize to integer multiple of 8
    mask = torch.nn.functional.interpolate(mask, (h // scale_factor, w // scale_factor))
    return mask

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False
    

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"

@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class InstructPix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()
        # import ipdb
        # ipdb.set_trace()
        if self.device.index:
            pipe.enable_model_cpu_offload(self.device.index)
        else:
            pipe.enable_model_cpu_offload(0)

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            print("Using full precision")
            pipe.unet.float()
            pipe.vae.float()

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae

        CONSOLE.print("InstructPix2Pix loaded!")

        self.resize512 = Resize((512,512))
        self.resize64 = Resize((64,64))


    def edit_image(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_cond: TensorType["BS", 3, "H", "W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 15,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98,
        mouth_rect=None,
        head_mask=None,
        face_aware=True,
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """
        # ipdb.set_trace()

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)

        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents

            
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

            mouth_rect = mouth_rect[0]

            mouth_image = self.resize512(image[:,:,mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]])
            mouth_image_cond = self.resize512(image_cond[:,:,mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]])

            mouth_rect_latent=(mouth_rect/512*64).to(dtype=torch.int32)
            mouth_latents = self.imgs_to_latent(mouth_image)
            mouth_image_cond_latents = self.prepare_image_latents(mouth_image_cond)


        # add noise
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0]) 

        noise = torch.randn_like(mouth_latents)
        mouth_latents = self.scheduler.add_noise(mouth_latents, noise, self.scheduler.timesteps[0])  

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # ipdb.set_trace()

                mouth_latent_model_input = torch.cat([mouth_latents] * 3)
                mouth_latent_model_input = torch.cat([mouth_latent_model_input, mouth_image_cond_latents], dim=1)

                mouth_noise_pred = self.unet(mouth_latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # ipdb.set_trace()
            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            mouth_noise_pred_text, mouth_noise_pred_image, mouth_noise_pred_uncond = mouth_noise_pred.chunk(3)
            mouth_noise_pred = (
                mouth_noise_pred_uncond
                + guidance_scale * (mouth_noise_pred_text - mouth_noise_pred_image)
                + image_guidance_scale * (mouth_noise_pred_image - mouth_noise_pred_uncond)
            )

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            mouth_latents = self.scheduler.step(mouth_noise_pred, t, mouth_latents).prev_sample


        # decode latents to get edited image
        with torch.no_grad():
            # ipdb.set_trace()
            decoded_img = self.latents_to_img(latents)
            mouth_decoded_img = self.latents_to_img(mouth_latents)
            decoded_img2 = decoded_img.clone()
            decoded_img2[:,:,mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]] = F.interpolate(mouth_decoded_img,mouth_rect[2])

            decoded_img3 = decoded_img2 * head_mask[:,:].unsqueeze(0).unsqueeze(0) + decoded_img* (1-head_mask[:,:].unsqueeze(0).unsqueeze(0))

        if face_aware == False:
            return decoded_img
        else:
            return decoded_img3

    def latents_to_img(self, latents: TensorType["BS", 4, "H", "W"]) -> TensorType["BS", 3, "H", "W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError






