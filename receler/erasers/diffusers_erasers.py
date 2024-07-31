import torch
import torch.nn as nn

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.lora import LoRACompatibleConv

from .utils import AdapterEraser


def diffuser_prefix_name(name):
    block_type = name.split('.')[0]
    if block_type == 'mid_block':
        return '.'.join(name.split('.')[:3])
    return  '.'.join(name.split('.')[:4])


def inject_eraser(unet, eraser_ckpt, eraser_rank, eraser_type='adapter'):
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlock):
            print(f'Load eraser at: {name}')
            prefix_name = diffuser_prefix_name(name)
            attn_w_eraser = AttentionWithEraser(module.attn2, eraser_rank)
            attn_w_eraser.adapter.load_state_dict(eraser_ckpt[prefix_name])
            module.attn2 = attn_w_eraser


class AttentionWithEraser(nn.Module):
    def __init__(self, attn, eraser_rank):
        super().__init__()
        self.attn = attn
        self.adapter = AdapterEraser(attn.to_out[0].weight.shape[1], eraser_rank)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        attn_outputs = self.attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        return self.adapter(attn_outputs) + attn_outputs
