import torch
import torch.nn.functional as F
import torch.nn as nn

class AdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, window_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.window_size = window_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(out_channels, ), requires_grad=True)

    def forward(self, x, l):
        # x: B C H W
        # loss: B 1 H W

        B, C, H, W = x.shape
        window_size = self.window_size
        kernel_size = self.kernel_size
        if l is None:
            gathered_inp = F.unfold(x, (kernel_size, kernel_size), padding=(kernel_size-1)//2)
        else:
            p = (window_size - 1) // 2
            padded_inp = F.pad(x, (p,p,p,p), 'reflect')
            l = F.pad(l, (p,p,p,p), 'constant', 999)
            l = F.unfold(l, (window_size, window_size))
            min_idx = torch.topk(l, k=kernel_size**2, dim=1, largest=False).indices
            unfold_inp = padded_inp.unfold(2, window_size, 1).unfold(3, window_size, 1).reshape(B, x.shape[1], H*W, window_size**2).permute(0, 1, 3, 2)
            gathered_inp = torch.gather(unfold_inp, dim=-2, index=min_idx.unsqueeze(1).repeat(1, x.shape[1], 1, 1)).view(B, -1, H*W)

        out_unf = gathered_inp.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        out = out_unf.view(B, -1, H, W) + self.bias.unsqueeze(0).repeat(B, 1).view(B, -1, 1, 1)
        return out, None