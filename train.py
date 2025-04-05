import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from SuperAD.model import SuperADTrainer
from dataset import plHADDataset
from name2dir import name2dir


def get_args_parser():
    parser = argparse.ArgumentParser('Training', add_help=False)
    parser.add_argument('--data_name', default="1_", type=str)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--rgb_c', default='9,5,3')
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--a', type=float, default=1)
    parser.add_argument('--b', type=float, default=1)
    parser.add_argument('--th_idx', type=float, default=0.25)
    parser.add_argument('--loss', default="OBPM")
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--window_size', default=5, type=int)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    model_name = "HAD." + SuperADTrainer.__name__

    output_dir = f"logs/{model_name}/log_d={args.data_name}_l={args.loss}_k={args.kernel_size}_w={args.window_size}_b={args.b}_a={args.a}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    seed_everything(args.seed)

    dataset = plHADDataset(name2dir[args.data_name])
    model = SuperADTrainer(
                      lr=args.lr,
                      epochs=args.epochs,
                      bands=dataset.dataset_train.bands,
                      rgb_c=[int(c) for c in args.rgb_c.split(",")],
                      data_name=args.data_name,
                      loss_name=args.loss,
                      kernel_size=args.kernel_size,
                      window_size=args.window_size,
                      alpha=args.a,
                      beta=args.b,
                      th_idx=args.th_idx,
                      )

    if args.wandb:
        wandb_logger = WandbLogger(project=model_name, name=output_dir, save_dir=output_dir)
    else:
        wandb_logger = CSVLogger(name=output_dir, save_dir=output_dir)

    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       monitor='AUC',
                                       mode="max",
                                       save_top_k=1,
                                       auto_insert_metric_name=False,
                                       filename='ep={epoch}_AUC={AUC:.6f}',
                                       save_last=True
                                       )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator="gpu",
                         devices=[0],
                         logger=wandb_logger,
                         callbacks=[model_checkpoint],
                         )

    trainer.fit(model, dataset)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

