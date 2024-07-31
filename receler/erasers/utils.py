import os
import json
import torch
import torch.nn as nn
from omegaconf import OmegaConf

def ldm_module_prefix_name(module_name):
    return '.'.join(module_name.split('.')[:2])

def save_eraser_to_diffusers_format(folder_path, erasers, eraser_rank, pretrained_cfg):
    # mapping from ldm to diffusers
    def ldm_to_diffusers(blk_prefix, lay_per_blk):
        blk_type, blk_id = blk_prefix.split('.')
        blk_id = int(blk_id)
        if blk_type == 'input_blocks':
            difs_blk_id = (blk_id - 1) // (lay_per_blk + 1)
            difs_lay_id = (blk_id - 1) % (lay_per_blk + 1)
            return f'down_blocks.{difs_blk_id}.attentions.{difs_lay_id}'
        elif blk_type == 'middle_block':
            return 'mid_block.attentions.0'
        elif blk_type == 'output_blocks':
            difs_blk_id = blk_id // (lay_per_blk + 1)
            difs_lay_id = blk_id % (lay_per_blk + 1)
            return f'up_blocks.{difs_blk_id}.attentions.{difs_lay_id}'
        else:
            raise AttributeError(f'Got unexpected block type: {blk_type}.')

    original_config = OmegaConf.load(pretrained_cfg)
    layers_per_block = original_config.model.params.unet_config.params['num_res_blocks']
    difs_eraser_ckpt = {}
    for eraser_name, eraser in erasers.items():
        eraser_blk_prefix = ldm_module_prefix_name(eraser_name)
        difs_blk_prefix = ldm_to_diffusers(eraser_blk_prefix, layers_per_block)
        difs_eraser_ckpt[difs_blk_prefix] = eraser.state_dict()

    # save eraser weights
    os.makedirs(folder_path, exist_ok=True)
    eraser_weight_path = os.path.join(folder_path, f"eraser_weights.pt")
    torch.save(difs_eraser_ckpt, eraser_weight_path)

    # save eraser config
    eraser_config = {
        'eraser_type': 'adapter',
        'eraser_rank': eraser_rank,
    }
    eraser_config_path = os.path.join(folder_path, "eraser_config.json")
    with open(eraser_config_path, 'w') as f:
        json.dump(eraser_config, f, indent=4)


class DisableEraser:
    def __init__(self, model, train=False):
        self.model = model
        self.train = train
        self.old_states = {}

    def __enter__(self):
        self.old_training = self.model.training
        self.model.train(self.train)
        # disable erasers
        for name, module in self.model.model.diffusion_model.named_modules():
            if isinstance(module, EraserControlMixin):
                self.old_states[name] = module.use_eraser
                module.use_eraser = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.train(self.old_training)
        # enable erasers
        for name, module in self.model.model.diffusion_model.named_modules():
            if isinstance(module, EraserControlMixin):
                module.use_eraser = self.old_states[name]


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class EraserControlMixin:
    _use_eraser = True

    @property
    def use_eraser(self):
        return self._use_eraser

    @use_eraser.setter
    def use_eraser(self, state):
        if not isinstance(state, bool):
            raise AttributeError(f'state should be bool, but got {type(state)}.')
        self._use_eraser = state


class AdapterEraser(nn.Module, EraserControlMixin):
    def __init__(self, dim, mid_dim):
        super().__init__()
        self.down = nn.Linear(dim, mid_dim)
        self.act = nn.GELU()
        self.up = zero_module(nn.Linear(mid_dim, dim))

    def forward(self, hidden_states):
        return self.up(self.act(self.down(hidden_states)))
