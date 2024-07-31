import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from einops import rearrange

from ldm.modules.attention import CrossAttention
from .erasers.utils import ldm_module_prefix_name


def get_mask(attn_maps, word_indices, thres):
    """
    attn_maps: {name: attns in shape (bs, heads, h*w, text_len)}
    word_indices: (num_tokens,)
    thres: float, threshold of mask
    """
    name2res = {}
    attns_choosen = []
    for name, attns in attn_maps.items():
        name = ldm_module_prefix_name(name)
        attns = attns[..., word_indices].mean(-1).mean(1)  # (bs, hw)
        res = int(np.sqrt(attns.shape[-1]))
        name2res[name] = res
        if res != 16:  # following MasaCtrl, we only use 16 x 16 cross attn maps
            continue
        attns = rearrange(attns, 'b (h w) -> b h w', h=res)  # (bs, h, w)
        attns_choosen.append(attns)
    # prepare mask
    attns_avg = torch.stack(attns_choosen, dim=1).mean(1)  # (bs, h, w)
    attn_min = attns_avg.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    attn_max = attns_avg.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    mask = (attns_avg - attn_min) / (attn_max - attn_min)  # normalize
    mask[mask >= thres] = 1
    mask[mask < thres] = 0
    # rescale mask for all possibility
    cached_masks = {}
    ret_masks = {}
    for name, res in name2res.items():
        if res in cached_masks:
            ret_masks[name] = cached_masks[res]
        else:
            rescaled_mask = F.interpolate(mask.unsqueeze(0), (res, res)).squeeze(0)
            cached_masks[res] = rescaled_mask
            ret_masks[name] = rescaled_mask
    return ret_masks


class AttnMapsCapture:
    def __init__(self, model, attn_maps):
        self.model = model
        self.attn_maps = attn_maps
        self.handlers = []

    def __enter__(self):
        for module_name, module in self.model.model.diffusion_model.named_modules():
            if 'transformer_blocks' in module_name and 'attn2' in module_name and isinstance(module, CrossAttention):
                handler = module.register_forward_hook(self.get_attn_maps(module_name))
                self.handlers.append(handler)

    def __exit__(self, exc_type, exc_value, traceback):
        for handler in self.handlers:
            handler.remove()

    def get_attn_maps(self, module_name):
            def hook(model, input, output):
                self.attn_maps[module_name] = output[1].detach()
            return hook


class EraserOutputsCapture:
    def __init__(self, model, erasers, eraser_outs):
        self.model = model
        self.eraser_names = list(erasers.keys())
        self.eraser_outs = eraser_outs
        self.handlers = []

    def __enter__(self):
        for module_name, module in self.model.model.diffusion_model.named_modules():
            if module_name in self.eraser_names:
                handler = module.register_forward_hook(self.get_eraser_outs(module_name))
                self.handlers.append(handler)

    def __exit__(self, exc_type, exc_value, traceback):
        for handler in self.handlers:
            handler.remove()

    def get_eraser_outs(self, module_name):
            def hook(model, input, output):
                self.eraser_outs[module_name] = output
            return hook
