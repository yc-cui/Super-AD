# Overcoming the Identity Mapping Problem in Self-Supervised Hyperspectral Anomaly Detection


This repository contains a training script for the SuperAD model, designed for self-supervised anomaly detection overcoming the identity mapping problem. The script utilizes PyTorch Lightning for efficient training and logging, and it supports integration with Weights & Biases (Wandb) for experiment tracking.



## Network Architecture

![](assets/overview.png)



## Requirements

```bash
conda env create -f environment.yml
conda activate HAD
```

## Training

```bash
python train.py --data_name <data_name>
```

The `data_name` can be modified in the `name2dir.py` file, you can also add your own data by modifying the `name2dir.py` file.

You can refer to `slic_viz.ipynb` for generating the SLIC superpixel for your own data.

The log file will be saved in the `logs` folder, we provide an example log file in the `logs/HAD.SuperADTrainer/log_d=1__l=OBPM_k=3_w=5_b=1_a=1` folder.

### Arguments

The script accepts the following command-line arguments, parameters may be tuned for different datasets to achieve better performance:

- --data_name: Name of the dataset to be used (default: “1_”).
- --epochs: Number of training epochs (default: 1000).
- --lr: Learning rate for the optimizer (default: 1e-3).
- --a: Alpha parameter for the model (default: 1).
- --b: Beta parameter for the model (default: 1).
- --kernel_size: Size of the convolutional kernel (default: 3).
- --window_size: Size of the sliding window (default: 5).
