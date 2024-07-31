import torch
import numpy as np
from pathlib import Path
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def get_pretrained_models(pretrained_cfg, pretrained_ckpt, device):

    model = load_model_from_config(pretrained_cfg, pretrained_ckpt, device)
    sampler = DDIMSampler(model)

    return model, sampler


def load_model_from_config(config, ckpt, device="cpu"):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, 
                 n_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
        
    log_t = log_every_t if log_every_t is not None else 100
    
    shape = [4, h // 8, w // 8]
    
    samples_ddim, inters = sampler.sample(
        S=ddim_steps,
        conditioning=c,
        batch_size=n_samples,
        shape=shape,
        verbose=False,
        x_T=start_code,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
        verbose_iter=verbose,
        t_start=t_start,
        log_every_t=log_t,
        till_T=till_T
    )
    
    if log_every_t is not None:
        return samples_ddim, inters
    
    return samples_ddim


def decode_latent_to_image(z, model):
    decoded_imgs = model.decode_first_stage(z)
    decoded_imgs = torch.clamp((decoded_imgs+1.0)/2.0, min=0.0, max=1.0)
    decoded_imgs = 255. * rearrange(decoded_imgs.cpu().numpy(), 'b c h w -> b h w c')
    images = [Image.fromarray(img.astype(np.uint8)) for img in decoded_imgs]
    return images