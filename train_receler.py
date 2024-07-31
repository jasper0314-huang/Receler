import os
import random
import shutil
import json
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf

from receler.erasers.utils import DisableEraser, save_eraser_to_diffusers_format
from receler.erasers.ldm_erasers import setup_ldm_adapter_eraser
from receler.convertModels import create_unet_diffusers_config
from receler.utils import sample_model, decode_latent_to_image, get_pretrained_models
from receler.concept_reg import ldm_module_prefix_name, get_mask, AttnMapsCapture, EraserOutputsCapture


def train_receler(
        args,
        concept,
        save_root,
        iterations=1000,
        lr=3e-4,
        start_guidance=3,
        negative_guidance=1,
        seperator=None,
        image_size=512,
        ddim_steps=50,
        pretrained_ckpt='./receler/sd-v1-4-full-ema.ckpt',
        pretrained_cfg='./receler/configs/stable-diffusion/v1-inference.yaml',
    ):
    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # extend specific concept
    word_print = concept.replace(' ', '')
    original_concept = concept

    concept_mappings = {
        'i2p': "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood",
    }

    concept = concept_mappings.get(original_concept, concept)

    # seperate concept string into (multiple) concepts
    if seperator is not None:
        words = concept.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [concept]
    ddim_eta = 0

    # get pretrained SDv1.4
    model, sampler = get_pretrained_models(pretrained_cfg, pretrained_ckpt, device)

    # setup eraser
    for param in model.parameters():
        param.requires_grad = False
    erasers = setup_ldm_adapter_eraser(model, eraser_rank=args.eraser_rank, device=device)

    # setup optimizer
    opt = torch.optim.Adam([param for eraser in erasers.values() for param in eraser.parameters()], lr=lr)

    # lambda function for only denoising till time step t
    quick_sample_till_t = lambda x, s, code, t: sample_model(
        model=model,
        sampler=sampler,
        c=x,
        h=image_size,
        w=image_size,
        ddim_steps=ddim_steps,
        scale=s,
        ddim_eta=ddim_eta,
        start_code=code,
        till_T=t,
        verbose=False
    )

    # setup experiment name
    erase_msg = f'rank_{args.eraser_rank}'
    advrs_msg = f'advrs_iter_{args.advrs_iters}-start_{args.start_advrs_train}-num_prompts_{args.num_advrs_prompts}'
    reg_msg = f'concept_reg_{args.concept_reg_weight}-mask_thres_{args.mask_thres}'
    name = f'receler-word_{word_print}-{erase_msg}-{advrs_msg}-{reg_msg}-iter_{iterations}-lr_{lr}'

    print('\n'.join(['#'*50, name, '#'*50]))

    folder_path = os.path.join(save_root, name)
    
    # dicts to store captured attention maps and eraser outputs
    attn_maps = {}
    eraser_outs = {}

    # create attack prompt embeddings
    if args.advrs_iters:
        advrs_prompt_embs = [nn.Parameter(torch.rand((1, args.num_advrs_prompts, 768), device=device))
                             for idx in range(len(words))]
        advrs_prompt_opts = [torch.optim.Adam([advrs_prompt_embs[idx]], lr=0.1, weight_decay=0.1)
                             for idx in range(len(words))]

    # training
    pbar = tqdm(range(iterations))
    for it in pbar:
        model.train()
        
        word_idx, word = random.sample(list(enumerate(words)),1)[0]
        # get text embeddings for unconditional and conditional prompts
        emb_0 = model.get_learned_conditioning([''])
        emb_p = model.get_learned_conditioning([f'{word}'])
        emb_n = model.get_learned_conditioning([f'{word}'])

        # hacking the indices of targeted word and adversarial prompts
        text_len = len(model.cond_stage_model.tokenizer(f'{word}', add_special_tokens=False)['input_ids'])
        word_indices = torch.arange(1, 1+text_len, device=device)
        advrs_indices = torch.arange(1+text_len, 1+text_len+args.num_advrs_prompts, device=device)

        # time step from 1000 to 0 (0 being good)
        t_enc = torch.randint(ddim_steps, (1,), device=device)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=device)

        start_code = torch.randn((1, 4, 64, 64)).to(device)

        with torch.no_grad():
            # generate an image with the concept from model
            z = quick_sample_till_t(emb_p.to(device), start_guidance, start_code, int(t_enc)) # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z
            with DisableEraser(model, train=False):
                e_0 = model.apply_model(z.to(device), t_enc_ddpm.to(device), emb_0.to(device))
                with AttnMapsCapture(model, attn_maps=attn_maps):
                    e_p = model.apply_model(z.to(device), t_enc_ddpm.to(device), emb_p.to(device))

        attn_masks = get_mask(attn_maps, word_indices, args.mask_thres)

        for inner_it in range(args.advrs_iters):

            # copy advrs_prompt_emb to input emb_n and make it requires_grad if advrs train
            emb_n = emb_n.detach()
            emb_n[:, advrs_indices, :].data = advrs_prompt_embs[word_idx].data
            emb_n.requires_grad = True

            # get conditional score from model
            with EraserOutputsCapture(model, erasers, eraser_outs):
                e_n = model.apply_model(z.to(device), t_enc_ddpm.to(device), emb_n.to(device))

            # perform advrs attack
            loss_adv = F.mse_loss(e_n, e_p, reduction='mean')
            tmp_grad, = torch.autograd.grad(loss_adv, [emb_n], retain_graph=True)
            advrs_prompt_embs[word_idx].grad = tmp_grad[:, advrs_indices, :]
            advrs_prompt_opts[word_idx].step()
            advrs_prompt_opts[word_idx].zero_grad()

            # perform erase training
            if inner_it == args.advrs_iters - 1:
                loss_total = torch.tensor(0.).to(device)
                e_0.requires_grad = False
                e_p.requires_grad = False
                loss_erase = F.mse_loss(e_n, e_0 - (negative_guidance * (e_p - e_0)))
                loss_total += loss_erase
                # compute cross attn regularization loss
                loss_eraser_reg = torch.tensor(0.).to(device)
                reg_count = 0
                for e_name, e_out in eraser_outs.items():
                    prefix_name = ldm_module_prefix_name(e_name)
                    if prefix_name not in attn_masks:
                        print(f'Warning: cannot compute regularization loss for {e_name}, because corresponding mask not found.')  # cannot find mask for regularizing
                        continue
                    reg_count += 1
                    mask = attn_masks[prefix_name]
                    flip_mask = (~mask.unsqueeze(1).bool()).float()  # (1, 1, w, h)
                    if e_out.dim() == 3:  # (1, w*h, dim) -> (1, dim, w, h)
                        w = flip_mask.shape[2]
                        e_out = rearrange(e_out, 'b (w h) d -> b d w h', w=w)
                    loss_eraser_reg += ((e_out * flip_mask) ** 2).mean(1).sum() / (flip_mask.sum() + 1e-9)
                loss_eraser_reg /= reg_count
                loss_total += args.concept_reg_weight * loss_eraser_reg

                # update weights to erase the concept
                loss_total.backward()
                opt.step()
                opt.zero_grad()
                pbar.set_postfix({"loss_total": loss_total.item()}, refresh=False)
                pbar.set_description_str(f"[{datetime.now().strftime('%H:%M:%S')}] Erase \"{concept}\"", refresh=False)

        # visualization
        if it == 0 or (it+1) % args.visualize_iters == 0:
            folder = os.path.join(os.path.join(save_root, f'{name}', 'visualize', f'iter_{it}'))
            os.makedirs(folder, exist_ok=True)
            model.eval()
            with torch.no_grad():
                for vis_idx in range(args.num_visualize):
                    vis_code = torch.randn((1, 4, 64, 64)).to(device)
                    vis_z = quick_sample_till_t(
                        emb_p.to(device),
                        start_guidance,
                        vis_code,
                        ddim_steps,
                    )
                    decode_latent_to_image(vis_z, model)[0].save(os.path.join(folder, f'{vis_idx}.png'))

        # save checkpoint
        if (it+1) % args.save_ckpt_iters == 0:
            model.eval()
            save_eraser_to_diffusers_format(
                os.path.join(folder_path, 'ckpts', f'iter_{it}'),
                erasers=erasers,
                eraser_rank=args.eraser_rank,
                pretrained_cfg=pretrained_cfg,
            )

    model.eval()
    save_eraser_to_diffusers_format(
        folder_path,  # save last ckpt directly under folder_path
        erasers=erasers,
        eraser_rank=args.eraser_rank,
        pretrained_cfg=pretrained_cfg,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument('--concept', help='Concept to erase. Multiple concepts can be separated by commas.', type=str, required=True)
    parser.add_argument('--save_root', help='root to save model checkpoint', type=str, required=False, default='./models/')
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=500)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=3e-4)

    # Arguments for cross-attention regularization
    parser.add_argument('--concept_reg_weight', help='weight of concept-localized regularization loss', type=float, default=0.1)
    parser.add_argument('--mask_thres', help='threshold to obtain cross-attention mask', type=float, default=0.1)

    # Arguments for adversarial training
    parser.add_argument('--advrs_iters', help='number of adversarial iterations', type=int, default=50)
    parser.add_argument('--start_advrs_train', help='iteration to start adversarial training', type=int, default=0)
    parser.add_argument('--num_advrs_prompts', help='number of attack prompts to add', type=int, default=16)

    # Save checkpoint and visualization arguments
    parser.add_argument('--save_ckpt_iters', help="save checkpoint every N iterations", type=int, default=100)
    parser.add_argument('--visualize_iters', help="generate images every N iterations", type=int, default=50)
    parser.add_argument('--num_visualize', help="number of images to visualize", type=int, default=3)

    # Other training configuration
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--eraser_rank', help='the rank of eraser', type=int, required=False, default=128)
    parser.add_argument('--pretrained_ckpt', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='./receler/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--pretrained_cfg', help='config path for stable diffusion v1-4', type=str, required=False, default='./receler/configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)

    args = parser.parse_args()

    train_receler(
        args,
        args.concept,
        args.save_root,
        iterations=args.iterations,
        lr=args.lr,
        start_guidance=args.start_guidance,
        negative_guidance=args.negative_guidance,
        seperator=args.seperator,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        pretrained_ckpt=args.pretrained_ckpt,
        pretrained_cfg=args.pretrained_cfg,
    )
