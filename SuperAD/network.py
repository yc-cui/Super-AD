import math
import torch
import torch.nn as nn
from typing import Tuple
from torch.nn import init
import torch.nn.functional as F
from torch_scatter import scatter
from timm.models.vision_transformer import Block
from SuperAD.AdaConv import AdaConv


class SequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class LeakyReLUWrapper(nn.LeakyReLU):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__(negative_slope=negative_slope, inplace=inplace)
    def forward(self, input1, input2=None):
        output1 = super().forward(input1)
        return output1, input2


class SuperADNetwork(nn.Module):
    def __init__(self, nch_in=189, nch_out=189, kernel_size=7, window_size=11):
        super(SuperADNetwork, self).__init__()
        in_channels = nch_in
        out_channels = nch_out
        dim = 32
        self.patch_embed = SequentialMultiInput(
            nn.Conv2d(in_channels, dim, 3, 1, 1, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect'),
        )
        embed_dim = dim
        num_heads = 1
        mlp_ratio = 2
        depth = 1
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        decoder_embed_dim = 32
        decoder_num_heads = 1
        decoder_depth = 1
        self.middle_conv = nn.Conv2d(dim, decoder_embed_dim, 3, stride=1, padding=1, padding_mode='reflect')
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.window_size = window_size
        self.kernel_size = kernel_size
        self.smconv = SequentialMultiInput(
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),       
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size)
        )
        self.output_block = SequentialMultiInput(
            AdaConv(decoder_embed_dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
        )

        self.post_conv = SequentialMultiInput(
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, out_channels, 1)
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, AdaConv):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)

    def forward(self, x, segs, err_map):
        # x: 1 C H W
        # segs 1 1 H W
        B, C, H, W = x.shape
        
        x = self.patch_embed(x)
        x_embed = x
        
        # 1 D H W
        segs_flat = segs.view(1, 1, -1) - 1
        x_flat = x.view(1, x.shape[1], -1)

        x_suppool = scatter(x_flat, segs_flat, dim=-1, reduce="mean").transpose(1, 2)
        # # 1 D S

        # ViT
        for blk in self.blocks:
            x_suppool = blk(x_suppool)
        x_suppool = self.norm(x_suppool)
        x_suppool = self.decoder_embed(x_suppool)

        for blk in self.decoder_blocks:
            x_suppool = blk(x_suppool)
            
        x_suppool = self.decoder_norm(x_suppool).transpose(1, 2)

        uppooled_x = torch.gather(x_suppool, dim=-1, index=segs_flat.repeat(1, x_suppool.shape[1], 1)).reshape(1, -1, H, W)

        x = self.middle_conv(x)

        if err_map is not None:
            out, l = self.smconv(x, err_map)
            x, l = self.output_block(out * uppooled_x, err_map)
            
        else:
            x, l = self.output_block(x * uppooled_x, err_map)

        return self.post_conv(x)
    
