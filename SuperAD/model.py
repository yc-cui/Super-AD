import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
import scipy.io as sio
from sorcery import dict_of
from thop import profile
from thop import clever_format


from utils import TensorToHSI, get_auc
from SuperAD.OBPM import OBPM
from SuperAD.network import SuperADNetwork

class SuperADTrainer(pl.LightningModule):
    def __init__(self, lr,
                 epochs,
                 bands,
                 rgb_c,
                 data_name,
                 loss_name,
                 kernel_size,
                 window_size,
                 alpha,
                 beta,
                 th_idx,
                 ):

        super().__init__()
        self.automatic_optimization = False

        self.rgb_c = rgb_c
        self.bands = bands
        self.data_name = data_name

        self.save_hyperparameters()
        self.model = SuperADNetwork(nch_in=self.bands, 
                                    nch_out=self.bands, 
                                    kernel_size=kernel_size, 
                                    window_size=window_size, 
                                    )

        self.loss_type = loss_name
        self.loss_alpha = alpha
        self.loss_beta = beta
        self.loss_th_idx = th_idx
        self.last_loss = None
        self.max_auc = 0.

    def configure_optimizers(self):
        lr = self.hparams.lr
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.5, 0.8), weight_decay=1e-4)
        # opt = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
        sche_opt = StepLR(opt, step_size=100, gamma=0.8)
        return [opt], [sche_opt]


    def forward(self, hsi, segs, err_map):
        pred = self.model(hsi, segs, err_map)
        out = dict_of(pred)
        return out

    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        if sche_pf is not None:
            sche_pf.step()

    def training_step(self, batch, batch_idx):
        hsi, gt, segs = batch["hsi"], batch["gt"], batch["segs"]
        if self.current_epoch < 1:
            out = self.forward(hsi, segs, self.last_loss)
        else:
            out = self.forward(hsi, segs, self.last_loss)
        pred = out["pred"]
        opt = self.optimizers()
        opt.zero_grad()
        loss = F.l1_loss(pred, hsi, reduction='none')
        total_loss, plt_loss, log_dict = OBPM(loss.mean(dim=1), 
                                              segs, 
                                              beta=self.loss_beta,
                                              alpha=self.loss_alpha,
                                              th_idx=self.loss_th_idx,
                                              loss_type=self.loss_type
                                              )
        self.manual_backward(total_loss)
        self.last_loss = loss.detach().mean(dim=1, keepdim=True)
        opt.step()

        hsi = TensorToHSI(hsi)
        pred = TensorToHSI(pred)
        auc, detectmap = get_auc(hsi, pred, gt.cpu().data.numpy())


        log_dict["AUC"] = auc
        log_dict["MAE"] = loss.mean().item()
        log_dict["lr"] = opt.param_groups[0]["lr"]
        log_dict["epoch"] = self.current_epoch


        if auc > self.max_auc:
            self.max_auc = auc
            print("\nAUC", self.max_auc, "\n")
            sio.savemat(os.path.join(self.logger.save_dir, f'max_auc_viz' + '.mat'), 
                    {
                    'recon':pred, 
                     "det": detectmap, 
                     "arloss": plt_loss.detach().cpu().numpy(), 
                     "auc": np.array([auc])
                     })
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)