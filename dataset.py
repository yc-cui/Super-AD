import torch
import numpy as np
import scipy.io as sio

import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sorcery import dict_of
import os

from utils import OPBS


class HADData(Dataset):
    def __init__(self, data_path, bs=False, nsegs=0):
        input_data = sio.loadmat(os.path.expanduser(data_path))
        image = input_data['data']
        image = image.astype(np.float32)

        if bs:
            bandidx = OPBS(image, 10)
            image = image[:, :, bandidx]

        segs = input_data['segs1'] if "segs1" in input_data.keys() else image
        if nsegs != 0:
            segs = input_data[f'segs{nsegs}']
        segs = segs.astype(np.float32)
        self.bands = image.shape[2]
        self.rows = image.shape[0]
        self.cols = image.shape[1]
        gt = input_data['map']
        self.gt = gt.astype(np.float32)
        image = ((image - image.min()) / (image.max() - image.min()))
        # train_data = np.expand_dims(image, axis=0)
        self.hsi = torch.from_numpy(image.transpose(2,0,1)).type(torch.FloatTensor)
        self.segs = torch.from_numpy(segs).type(torch.int64).unsqueeze(0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        hsi = self.hsi
        gt = self.gt
        segs = self.segs
        inp_dict = dict_of(hsi, gt, segs)

        return inp_dict


class plHADDataset(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=1, num_workers=1, pin_memory=True, bs=False, nsegs=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_train = HADData(data_path, bs=bs, nsegs=nsegs)
        

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None