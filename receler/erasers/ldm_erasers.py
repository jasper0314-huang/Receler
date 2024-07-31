import torch
import torch.nn as nn

from .utils import AdapterEraser
from ldm.modules.attention import BasicTransformerBlock, GEGLU


class BasicTransformerBlockWithEraser(BasicTransformerBlock):
    def __init__(self, dim, n_heads, d_head, eraser_rank, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__(dim, n_heads, d_head, dropout, context_dim,
                         gated_ff, checkpoint, disable_self_attn)

        self.adapter = AdapterEraser(dim, eraser_rank)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None)[0] + x
        if self.adapter.use_eraser:
            ca_output = self.attn2(self.norm2(x), context=context)[0]
            x = self.adapter(ca_output) + ca_output + x
        else:
            x = self.attn2(self.norm2(x), context=context)[0] + x
        x = self.ff(self.norm3(x)) + x
        return x

    @classmethod
    def from_pretrained_block(cls, block, eraser_rank):
        dim = block.norm1.weight.shape[0]
        n_heads = block.attn1.heads
        d_head = round(block.attn1.scale ** -2)
        dropout = block.attn1.to_out[1].p
        context_dim = block.context_dim
        gated_ff = isinstance(block.ff.net[0], GEGLU)
        checkpoint = block.checkpoint
        disable_self_attn = block.disable_self_attn
        block_w_adapter = cls(dim, n_heads, d_head, eraser_rank, dropout,
                              context_dim, gated_ff, checkpoint, disable_self_attn)
        block_w_adapter.load_state_dict(block.state_dict(), strict=False)
        return block_w_adapter


def setup_ldm_adapter_eraser(model, eraser_rank, device):
    def replace_transformer_block(module):
        for name, child in module.named_children():
            if isinstance(child, BasicTransformerBlock):
                block_w_adapter = BasicTransformerBlockWithEraser.from_pretrained_block(child, eraser_rank).to(device)
                setattr(module, name, block_w_adapter)
            else:
                replace_transformer_block(child)
    replace_transformer_block(model)
    erasers = {}
    for name, module in model.model.diffusion_model.named_modules():
        if isinstance(module, BasicTransformerBlockWithEraser):
            eraser_name = f'{name}.adapter'
            print(eraser_name)
            erasers[eraser_name] = module.adapter
    return erasers
