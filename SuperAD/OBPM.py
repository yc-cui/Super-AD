import torch
import torch.nn.functional as F


def OBPM(loss, segs, beta, alpha, th_idx, loss_type, split="train",):
    if loss_type == "l1":
        loss = loss
    elif loss_type == "l2":
        loss = loss * loss
    else:
        loss = torch.exp(beta * loss) / beta + loss * alpha
    plt_loss = loss.sum(dim=1, keepdim=False)

    # x: B H W
    # bool_mask: B 1 H W
    N = segs.max().item()   
    expanded_mask = segs.expand(-1, N, -1, -1)  # 形状变为[B, N, H, W]
    bool_mask = (expanded_mask == torch.arange(1, N+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(loss.device))  # 形状为[1, N, 1, 1]  
    B, H, W = loss.shape
    select_loss = loss.unsqueeze(1) * bool_mask.float()  # 形状变为[B, N, H, W]
    flat_select_loss = select_loss.reshape(B, N, -1)  # 形状变为[B, N, H*W]
    min_val = torch.where(bool_mask.reshape(B, N, -1), flat_select_loss,torch.inf).min(-1).values
    is_inf = torch.isinf(min_val)
    min_val = torch.where(is_inf, torch.tensor(0.0), min_val)

    sorted_flat_select_loss, _ = flat_select_loss.sort(dim=-1)  # 转置并排序
    mask = sorted_flat_select_loss == 0
    min_val_expanded = min_val.unsqueeze(-1).expand_as(sorted_flat_select_loss)
    sorted_flat_select_loss.masked_scatter_(mask, min_val_expanded[mask])

    diff1 = torch.diff(sorted_flat_select_loss, dim=-1, prepend=sorted_flat_select_loss.permute(2,0,1)[0].unsqueeze(-1))

    K = 7
    kernel = torch.ones(1, 1, K).to(loss.device) / K
    acc_diff1 = F.conv1d(diff1.view(-1, 1, diff1.shape[-1]), kernel, stride=1, padding=(K-1)//2).view(diff1.shape)

    max_idx = acc_diff1.argmax(dim=-1)
    th_loss = sorted_flat_select_loss.gather(-1, max_idx.unsqueeze(-1))

    ignore_idx = (H*W - max_idx) >= (bool_mask.sum((-1, -2)) * (1 - th_idx))
    th_loss[ignore_idx] = 1e4   
    loss = torch.where(flat_select_loss <= th_loss, flat_select_loss, torch.tensor(0.0).to(loss.device))
    plt_loss = loss.sum(dim=1, keepdim=False).reshape(H, W)

    total_loss = plt_loss.mean()
    log_dict = {
        f"{split}/total_loss": total_loss
    }

    return total_loss, plt_loss, log_dict