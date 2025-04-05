import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import random
from torch.nn import init

def map01(img):
    img_01 = (img - img.min())/(img.max() - img.min() + 1e-8)
    return img_01

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_auc_gt_err(gt, detectmap, norm=True):
    # print(gt.shape,detectmap.shape)
    n_row, n_col = gt.shape[0], gt.shape[1]
    n_pixels = n_row * n_col
        
    # nomalization
    if norm:
        detectmap = map01(detectmap)

    # get auc
    label = np.reshape(gt, (n_pixels,1), order='F')
    detectmap = np.reshape(detectmap, (n_pixels,1), order='F')
    
    detectmap = np.nan_to_num(detectmap)
    auc = roc_auc_score(label, detectmap)
    
    detectmap = np.reshape(detectmap, (n_row, n_col), order='F')
    
    return auc, detectmap

def get_auc(HSI_old, HSI_new, gt):
    n_row, n_col, n_band = HSI_old.shape
    n_pixels = n_row * n_col
        
    img_olds = np.reshape(HSI_old, (n_pixels, n_band), order='F')
    img_news = np.reshape(HSI_new, (n_pixels, n_band), order='F')        
    sub_img = img_olds - img_news

    detectmap = np.linalg.norm(sub_img, ord = 2, axis = 1, keepdims = True)**2
    detectmap = detectmap/n_band

    # nomalization
    detectmap = map01(detectmap)

    # get auc
    label = np.reshape(gt, (n_pixels,1), order='F')
    try:
        auc = roc_auc_score(label, detectmap)
    except:
        auc = 0.
    detectmap = np.reshape(detectmap, (n_row, n_col), order='F')
    
    return auc, detectmap

def TensorToHSI(img):
    HSI = img.squeeze().cpu().data.numpy().transpose((1, 2, 0))
    return HSI

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



import numpy as np


def OPBS(
        x: np.ndarray,
        num_bs: int
):
    """
    Ref:
    W. Zhang, X. Li, Y. Dou, and L. Zhao, “A geometry-based band
    selection approach for hyperspectral image analysis,” IEEE Transactions
    on Geoscience and Remote Sensing, vol. 56, no. 8, pp. 4318–4333, 2018.
    """
    rows, cols, bands = x.shape
    eps = 1e-9

    x_2d = np.reshape(x, (rows * cols, bands))
    y_2d = x_2d.copy()
    h = np.zeros(bands)
    band_idx = []

    idx = np.argmax(np.var(x_2d, axis=0))
    band_idx.append(idx)
    h[idx] = np.sum(x_2d[:, band_idx[-1]] ** 2)

    i = 1
    while i < num_bs:
        id_i_1 = band_idx[i - 1]

        _elem, _idx = -np.inf, 0
        for t in range(bands):
            if t not in band_idx:
                y_2d[:, t] = y_2d[:, t] - y_2d[:, id_i_1] * (np.dot(y_2d[:, id_i_1], y_2d[:, t]) / (h[id_i_1] + eps))
                h[t] = np.dot(y_2d[:, t], y_2d[:, t])

                if h[t] > _elem:
                    _elem = h[t]
                    _idx = t

        band_idx.append(_idx)
        i += 1

    band_idx = sorted(band_idx)
    return band_idx
