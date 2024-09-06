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

# try:
#     from diffusers import (
#         DDIMScheduler,
#         ControlNetModel,
#         StableDiffusionControlNetPipeline,
#         StableDiffusionInstructPix2PixPipeline,
#     )
#     from transformers import logging

# except ImportError:
#     CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
#     CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
#     CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
#     sys.exit(1)

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

# from .ext_attn import AttnProcessor2_0

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

class FinetuneModel(nn.Module):
    def __init__(self,
                 device: Union[torch.device, str],
                 dtype = torch.float16,
                 num_train_timesteps: int = 1000,
                 use_full_precision = False,
                 base_path: str = None,
                 controlnet_path: str = None
                 ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_train_timesteps = num_train_timesteps
        self.use_full_precision = use_full_precision
        
        # controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=self.dtype, cache_dir=diffusion_cache_dir)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(base_path, controlnet=controlnet, torch_dtype=self.dtype, safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_pretrained(base_path, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)
        self.pipe = pipe
        # improve memory performance
        # pipe.enable_attention_slicing()
        # if pipe.device.index:
        #     pipe.enable_model_cpu_offload(pipe.device.index)
        # else:
        #     pipe.enable_model_cpu_offload(0)
            
        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore
        pipe.unet.eval()
        pipe.vae.eval()
        
        # use for improved quality at cost of higher memory
        if self.use_full_precision:
            print("Using full precision")
            pipe.unet.float()
            pipe.vae.float()

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae
        self.vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)

        CONSOLE.print("Finetune Model loaded!")
    
    def train_step(
        self,
        prompt_embeds,
        image: TensorType["BS", 3, "H", "W"], ## NeRF/Gaussian Render Image,
        control_image: TensorType["BS", 3, "H", "W"] = None,
        mask_image: TensorType["BS", 3, "H", "W"] = None,
        grad_scale=1,
        guidance_scale: float = 7.5,
        diffusion_steps: int = 40,
        lower_bound: float = 0.50,
        upper_bound: float = 0.98,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        T_ratio: float = None,
        sds_output = True,
        image_output = True
    ):
        (bs, _, hei, wid) = image.shape
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)
        do_classifier_free_guidance = guidance_scale > 1.0
        
        T = torch.randint(min_step, max_step + 1, (image.shape[0],), dtype=torch.long, device=self.device)
        if T_ratio is not None:
            T = torch.tensor([round(T_ratio * self.num_train_timesteps)], dtype=torch.long, device=self.device)
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)
        
        alphas = self.scheduler.alphas_cumprod.to(self.device)
        
        # align format for control guidance
        control_guidance_start, control_guidance_end =  [control_guidance_start], [control_guidance_end]
        controlnet_keep = []
        for i in range(len(self.scheduler.timesteps)):
            keeps = [
                1.0 - float(i / len(self.scheduler.timesteps) < s or (i + 1) / len(self.scheduler.timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0])
        controlnet_cond_scale = controlnet_conditioning_scale
        cond_scale = controlnet_cond_scale * controlnet_keep[i]
        
        assert (control_image is not None), 'Openpose condition not found!'
        with torch.no_grad():
            control_image = self.pipe.prepare_image(
                image=control_image,
                width=wid,
                height=hei,
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=self.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=False,
            )
        # prepare image and image_cond latents
        latents = self.imgs_to_latent(image)
        with torch.no_grad():
            #prepare mask
            mask_image = preprocess_mask(mask_image, bs, self.vae_scale_factor)
            mask = mask_image.to(device=self.device, dtype=torch.float16)
            mask = torch.cat([mask] * 1)
        
            noise = torch.randn_like(latents)
        if sds_output:
            with torch.no_grad():
                latents_noisy = self.scheduler.add_noise(latents, noise, T)  # type: ignore
                latent_model_input = torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents_noisy
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

        
                # import ipdb; ipdb.set_trace()
                # a = time.time()
            
                tt = torch.cat([T] * 2)
                down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    control_model_input,
                    tt,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    guess_mode=False,
                    return_dict=False,
                )
                
                noise_pred = self.pipe.unet(
                        latent_model_input,
                        tt,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                w = (1 - alphas[T])
                grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
                grad = torch.nan_to_num(grad)
                
                targets = (latents - grad).detach()
            loss = 0.5 * F.mse_loss((latents*mask).float(), targets*mask, reduction='sum') / latents.shape[0]
            # b = time.time()
            # print(b-a)
        else:
            loss = None
        
        if image_output:
            with torch.no_grad():
                new_latent = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore
                for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                    latent_model_input = torch.cat([new_latent] * 2) if do_classifier_free_guidance else new_latent
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    
                    down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        return_dict=False,
                    )
                    noise_pred = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # get previous sample, continue loop
                    new_latent = self.scheduler.step(noise_pred, t, new_latent).prev_sample
                # decode latents to get edited image
                decoded_img = self.latents_to_img(new_latent)
        else:
            decoded_img = None
            
        return loss, decoded_img
    
    def train_one_step(
        self,
        prompt_embeds,
        image: TensorType["BS", 3, "H", "W"], ## NeRF/Gaussian Render Image,
        control_image: TensorType["BS", 3, "H", "W"] = None,
        mask_image: TensorType["BS", 3, "H", "W"] = None,
        grad_scale=1,
        guidance_scale: float = 7.5,
        diffusion_steps: int = 40,
        lower_bound: float = 0.50,
        upper_bound: float = 0.98,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        T_ratio: float = None,
        sds_output = True,
        image_output = True
    ):
        (bs, _, hei, wid) = image.shape
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)
        do_classifier_free_guidance = guidance_scale > 1.0
        
        T = torch.randint(min_step, max_step + 1, (image.shape[0],), dtype=torch.long, device=self.device)
        if T_ratio is not None:
            T = torch.tensor([round(T_ratio * self.num_train_timesteps)], dtype=torch.long, device=self.device)
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)
        
        alphas = self.scheduler.alphas_cumprod.to(self.device)
        
        # align format for control guidance
        control_guidance_start, control_guidance_end =  [control_guidance_start], [control_guidance_end]
        controlnet_keep = []
        for i in range(len(self.scheduler.timesteps)):
            keeps = [
                1.0 - float(i / len(self.scheduler.timesteps) < s or (i + 1) / len(self.scheduler.timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0])
        controlnet_cond_scale = controlnet_conditioning_scale
        cond_scale = controlnet_cond_scale * controlnet_keep[i]
        assert (control_image is not None), 'Openpose condition not found!'
        with torch.no_grad():
            control_image = self.pipe.prepare_image(
                image=control_image,
                width=wid,
                height=hei,
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=self.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=False,
            )
        # prepare image and image_cond latents
        latents = self.imgs_to_latent(image)
        with torch.no_grad():
            #prepare mask
            mask_image = preprocess_mask(mask_image, bs, self.vae_scale_factor)
            mask = mask_image.to(device=self.device, dtype=torch.float16)
            mask = torch.cat([mask] * 1)
        
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, T)  # type: ignore
            
            latent_model_input = torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents_noisy
            control_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds

            # import ipdb; ipdb.set_trace()
            # a = time.time()
        with torch.no_grad():
            tt = torch.cat([T] * 2)
            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                control_model_input,
                tt,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=control_image,
                conditioning_scale=cond_scale,
                guess_mode=False,
                return_dict=False,
            )
            
            noise_pred = self.pipe.unet(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            w = (1 - alphas[T])
            grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
            grad = torch.nan_to_num(grad)
            
            targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss((latents*mask).float(), targets*mask, reduction='sum') / latents.shape[0]
        # b = time.time()
        # print(b-a)
        
        # total_timesteps = max_step - min_step + 1
        # index = total_timesteps - T.to(latents.device) - 1 
        index = T.to(latents.device)
        b = len(noise_pred)
        a_t = alphas[index].reshape(b,1,1,1).to(self.device)
        sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
        pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
        # result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))
        decoded_img = self.latents_to_img(pred_x0.to(latents.type(self.dtype)))
            
        return loss, decoded_img
    
    def train_step_inpainting(
        self,
        prompt_embeds,
        image: TensorType["BS", 3, "H", "W"], ## NeRF/Gaussian Render Image,
        control_image: TensorType["BS", 3, "H", "W"] = None,
        mask_image: TensorType["BS", 3, "H", "W"] = None,
        grad_scale=1,
        guidance_scale: float = 7.5,
        diffusion_steps: int = 40,
        lower_bound: float = 0.50,
        upper_bound: float = 0.98,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        T_ratio: float = None,
        sds_output = True,
        image_output = True
    ):
        (bs, _, hei, wid) = image.shape
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)
        do_classifier_free_guidance = guidance_scale > 1.0
        
        T = torch.randint(min_step, max_step + 1, (image.shape[0],), dtype=torch.long, device=self.device)
        if T_ratio is not None:
            T = torch.tensor([round(T_ratio * self.num_train_timesteps)], dtype=torch.long, device=self.device)
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)
        
        alphas = self.scheduler.alphas_cumprod.to(self.device)
        
        # align format for control guidance
        control_guidance_start, control_guidance_end =  [control_guidance_start], [control_guidance_end]
        controlnet_keep = []
        for i in range(len(self.scheduler.timesteps)):
            keeps = [
                1.0 - float(i / len(self.scheduler.timesteps) < s or (i + 1) / len(self.scheduler.timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0])
        controlnet_cond_scale = controlnet_conditioning_scale
        cond_scale = controlnet_cond_scale * controlnet_keep[i]
        
        assert (control_image is not None), 'Openpose condition not found!'
        with torch.no_grad():
            control_image = self.pipe.prepare_image(
                image=control_image,
                width=wid,
                height=hei,
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=self.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=False,
            )
        # prepare image and image_cond latents
        latents = self.imgs_to_latent(image)
        with torch.no_grad():
            #prepare mask
            mask_image = preprocess_mask(mask_image, bs, self.vae_scale_factor)
            mask = mask_image.to(device=self.device, dtype=torch.float16)
            mask = torch.cat([mask] * 1)
        
            noise = torch.randn_like(latents)
        if sds_output:
            with torch.no_grad():
                latents_noisy = self.scheduler.add_noise(latents, noise, T)  # type: ignore
                latent_model_input = torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents_noisy
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

        
                # import ipdb; ipdb.set_trace()
                # a = time.time()
            
                tt = torch.cat([T] * 2)
                down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    control_model_input,
                    tt,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    guess_mode=False,
                    return_dict=False,
                )
                
                noise_pred = self.pipe.unet(
                        latent_model_input,
                        tt,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                w = (1 - alphas[T])
                grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
                grad = torch.nan_to_num(grad)
                
                targets = (latents - grad).detach()
            loss = 0.5 * F.mse_loss((latents*mask).float(), targets*mask, reduction='sum') / latents.shape[0]
            # b = time.time()
            # print(b-a)
        else:
            loss = None
        
        if image_output:
            with torch.no_grad():
                init_latents_orig = latents
                new_latent = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore
                for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                    latent_model_input = torch.cat([new_latent] * 2) if do_classifier_free_guidance else new_latent
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    
                    down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        return_dict=False,
                    )
                    noise_pred = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # get previous sample, continue loop
                    new_latent = self.scheduler.step(noise_pred, t, new_latent).prev_sample
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise_pred_uncond, torch.tensor([t]))
                    new_latent = (init_latents_proper * (1 - mask)) + (new_latent * mask)
                # decode latents to get edited image
                decoded_img = self.latents_to_img(new_latent)
        else:
            decoded_img = None
            
        return loss, decoded_img
    
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

        # for _, module in self.pipe.unet.named_modules():
        #     if isinstance_str(module, "BasicTransformerBlock"):
        #         # ipdb.set_trace()
        #         # print(type(module.attn1))
        #         module.attn1.set_processor(AttnProcessor2_0(0.0))
        # CONSOLE.print("EXT ATTN loaded!")

    def edit_image(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_cond: TensorType["BS", 3, "H", "W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
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
            # ipdb.set_trace()
            
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

            mouth_rect = mouth_rect[0]

            mouth_image = self.resize512(image[:,:,mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]])
            mouth_image_cond = self.resize512(image_cond[:,:,mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]])

            # ipdb.set_trace()
            mouth_rect_latent=(mouth_rect/512*64).to(dtype=torch.int32)
            mouth_latents = self.imgs_to_latent(mouth_image)
            mouth_image_cond_latents = self.prepare_image_latents(mouth_image_cond)
            # ipdb.set_trace()


        # add noise
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore

        noise = torch.randn_like(mouth_latents)
        mouth_latents = self.scheduler.add_noise(mouth_latents, noise, self.scheduler.timesteps[0])  # type: ignore

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

                #paste back
                # noise_pred[:,:,mouth_rect_latent[1]:mouth_rect_latent[1]+mouth_rect_latent[3], mouth_rect_latent[0]:mouth_rect_latent[0]+mouth_rect_latent[2]]=F.interpolate(mouth_noise_pred,mouth_rect_latent[2])

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

        # latents[:,:,mouth_rect_latent[1]:mouth_rect_latent[1]+mouth_rect_latent[3], mouth_rect_latent[0]:mouth_rect_latent[0]+mouth_rect_latent[2]] = F.interpolate(mouth_latents,mouth_rect_latent[2])

        # decode latents to get edited image
        with torch.no_grad():
            # ipdb.set_trace()
            decoded_img = self.latents_to_img(latents)
            mouth_decoded_img = self.latents_to_img(mouth_latents)
            decoded_img2 = decoded_img.clone()
            decoded_img2[:,:,mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]] = F.interpolate(mouth_decoded_img,mouth_rect[2])

            decoded_img3 = decoded_img2 * head_mask[:,:].unsqueeze(0).unsqueeze(0) + decoded_img* (1-head_mask[:,:].unsqueeze(0).unsqueeze(0))

            # np_mouth_image=torch.clamp(mouth_decoded_img[0].permute(1,2,0).detach()*255, 0, 255).byte().cpu().numpy()
            # cv2.imwrite("debug_mouth.jpg",np_mouth_image[:,:,[2,1,0]])

            # np_image=torch.clamp(decoded_img[0].permute(1,2,0).detach()*255, 0, 255).byte().cpu().numpy()
            # cv2.imwrite("debug.jpg",np_image[:,:,[2,1,0]])

            # np_image_com=torch.clamp(decoded_img3[0].permute(1,2,0).detach()*255, 0, 255).byte().cpu().numpy()
            # cv2.imwrite("debug_com.jpg",np_image_com[:,:,[2,1,0]])

            # head_mask_image=torch.clamp(head_mask[:,:].unsqueeze(0).permute(1,2,0).detach()*255, 0, 255).byte().cpu().numpy()
            # cv2.imwrite("debug_head_mask.jpg",head_mask_image[:,:])

            # np_mouth_gt_image=torch.clamp(mouth_image_cond[0].permute(1,2,0).detach()*255, 0, 255).byte().cpu().numpy()
            # cv2.imwrite("debug_mouth_gt.jpg",np_mouth_gt_image[:,:,[2,1,0]])

            # np_gt_image=torch.clamp(image_cond[0].permute(1,2,0).detach()*255, 0, 255).byte().cpu().numpy()
            # cv2.imwrite("debug_gt.jpg",np_gt_image[:,:,[2,1,0]])

            # ipdb.set_trace()

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


if __name__ == "__main__":
    print('test')
    import cv2
    import numpy as np
    device = "cuda:0"
    devi = torch.device(0)
    
    bs = 10

    ip2p_module = InstructPix2Pix(devi, ip2p_use_full_precision=False)
    # self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)
    text_embedding = ip2p_module.pipe._encode_prompt(
                "give him a mustache", device=ip2p_module.device, num_images_per_prompt=bs, do_classifier_free_guidance=True, negative_prompt=""
            )
    print(text_embedding.shape)
    img_path = 'test/out900+900.jpg'
    gt_img_path = 'test/test.jpg'

    img = cv2.imread(img_path)
    gt_img = cv2.imread(gt_img_path)
    # cv2.imwrite('test/test.jpg', img[:,:512])
    # input('cont :')
    img_in = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).to(device) / 255.
    gt_img = torch.tensor(gt_img).unsqueeze(0).permute(0,3,1,2).to(device) / 255.
    img_in = img_in.expand(bs,3,512,512)
    gt_img = gt_img.expand(bs,3,512,512)

    print(img_in.shape)

    print(text_embedding.type(torch.float16).shape,
                img_in.clone().type(torch.float16).shape,
                gt_img.clone().type(torch.float16).shape,)

    ts = time.time()
    # gt_img = img_in
    with torch.no_grad():
        edited_image = ip2p_module.edit_image(
                text_embedding.type(torch.float16),
                img_in.clone().type(torch.float16),
                gt_img.clone().type(torch.float16),
                guidance_scale=12,
                image_guidance_scale=2.,
                diffusion_steps=20,
                lower_bound=0.90,
                upper_bound=0.91,
            )
    img_in = edited_image
    te = time.time()

    out_image = torch.clip(edited_image.detach()[(bs-1)].permute(1,2,0)*255.,0,255)
    cv2.imwrite('test/out9900.jpg', out_image.cpu().numpy().astype(np.float32))
    print('done, time : ', te - ts)








