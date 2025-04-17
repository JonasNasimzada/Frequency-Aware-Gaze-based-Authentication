# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.datamodules import GazeBaseDataModule, JuDo1000DataModule
from src.models.modules import EyeKnowYouToo

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    choices=["train", "test"],
    required=True,
    help="Whether to train or test a model",
)
parser.add_argument(
    "--resume_epoch",
    default=-1,
    type=int,
    help="If resuming training from a checkpoint, the number in 'epoch=___'",
)
parser.add_argument(
    "--fold",
    default=0,
    type=int,
    choices=[0, 1, 2, 3],
    help="The fold to use as the validation set.  Must train one model per fold to enable evaluation.",
)
parser.add_argument(
    "--map_at_r",
    action="store_true",
    help="Flag indicating to compute MAP@R while training",
)
parser.add_argument(
    "--w_ms", default=1.0, type=float, help="Weight for multi-similarity loss"
)
parser.add_argument(
    "--w_ce", default=0.1, type=float, help="Weight for cross-entropy loss"
)
parser.add_argument(
    "--gazebase_dir",
    default="./data/spectrogram_plots",
    type=str,
    help="Path to directory to store GazeBase data files",
)
parser.add_argument(
    "--judo_dir",
    default="./data/judo1000",
    type=str,
    help="Path to directory to store JuDo1000 data files",
)
parser.add_argument(
    "--log_dir",
    default="./lightning_logs",
    type=str,
    help="Path to directory to store Tensorboard logs",
)
parser.add_argument(
    "--ckpt_dir",
    default="./models",
    type=str,
    help="Path to directory to store model checkpoints",
)
parser.add_argument(
    "--embed_dir",
    default="./embeddings",
    type=str,
    help="Path to directory to store embeddings",
)
parser.add_argument(
    "--seq_len",
    default=5000,
    type=int,
    help="Length of input sequences (prior to downsampling)",
)
parser.add_argument(
    "--batch_classes",
    default=16,
    type=int,
    help="Number of classes sampled per minibatch",
)
parser.add_argument(
    "--batch_samples",
    default=16,
    type=int,
    help="Number of sequences sampled per class per minibatch",
)
parser.add_argument(
    "--ds",
    default=1,
    type=int,
    choices=[1, 2, 4, 8, 20, 32],
    help="Downsample factor.  Supported factors are 1 (1000 Hz), 2 (500 Hz), 4 (250 Hz), 8 (125 Hz), 20 (50 Hz), or 32 (31.25 Hz).",
)
parser.add_argument(
    "--gpu",
    default=0,
    type=int,
    help="The index of the GPU to use (based on order in 'nvidia-smi')",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Flag indicating to use the CPU instead of a GPU",
)
parser.add_argument(
    "--batch_size_for_testing",
    default=-1,
    type=int,
    help="Override the batch size with this value when `--mode=test`",
)
parser.add_argument(
    "--degrade_precision",
    action="store_true",
    help="Flag indicating to degrade spatial precision by adding white noise with SD=0.5 deg",
)
args = parser.parse_args()

# Hide all GPUs except the one we (maybe) want to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = "cpu" if args.cpu or not torch.cuda.is_available() else "gpu"

checkpoint_stem = (
        "ekyt"
        + f"_t{args.seq_len}"
        + f"_ds{args.ds}"
        + f"_bc{args.batch_classes}"
        + f"_bs{args.batch_samples}"
        + f"_wms{round(10.0 * args.w_ms):02d}"
        + f"_wce{round(10.0 * args.w_ce):02d}"
        + ("_degraded" if args.degrade_precision else "_normal")
        + f"_f{args.fold}"
)
checkpoint_path = Path(args.ckpt_dir) / (checkpoint_stem + ".ckpt")

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path.parent,
    monitor="val_metric_loss",
    filename=checkpoint_stem,
    save_top_k=1,
    mode="min",
    verbose=True
)
downsample_factors_dict = {
    1: [],
    2: [2],
    4: [4],
    8: [8],
    20: [4, 5],
    32: [8, 4],
}
downsample_factors = downsample_factors_dict[args.ds]

noise_sd = None
if args.degrade_precision:
    noise_sd = 0.5

test_batch_size = args.batch_size_for_testing
if test_batch_size == -1:
    test_batch_size = None
gazebase = GazeBaseDataModule(
    current_fold=args.fold,
    base_dir=args.gazebase_dir,
    downsample_factors=downsample_factors,
    subsequence_length_before_downsampling=args.seq_len,
    classes_per_batch=args.batch_classes,
    samples_per_class=args.batch_samples,
    compute_map_at_r=args.map_at_r,
    batch_size_for_testing=test_batch_size,
    noise_sd=noise_sd,
)

# Prepare field `n_classes` for model and fields `zscore_mn` and
# PyTorch Lightning modules, and it performs redundant data processing.
gazebase.prepare_data()
gazebase.setup(stage="fit")

model = EyeKnowYouToo(
    n_classes=gazebase.n_classes,
    embeddings_filename=checkpoint_stem + ".csv",
    embeddings_dir=args.embed_dir,
    w_metric_loss=args.w_ms,
    w_class_loss=args.w_ce,
    compute_map_at_r=args.map_at_r,
)

logger = None
if args.mode == "train":
    # We used the WandbLogger during our experimentation, but we use the
    # TensorBoardLogger in this public release for simplicity
    logger = TensorBoardLogger(save_dir=args.log_dir, name=checkpoint_stem)

trainer = pl.Trainer(
    devices=1,
    accelerator=device,
    callbacks=[
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        checkpoint_callback,
    ],
    logger=logger,
    fast_dev_run=False,
    max_epochs=100,
    log_every_n_steps=50,
    precision=32,
    benchmark=False,
    deterministic=False,
    auto_lr_find=False,
    detect_anomaly=True,
)

if args.mode == "train":
    epoch = args.resume_epoch
    ckpt = None
    if epoch != -1:
        ckpt = str(
            checkpoint_path.with_name(checkpoint_stem)
        )
    trainer.fit(model, gazebase, ckpt_path=ckpt)
elif args.mode == "test":
    ckpt = str(checkpoint_path.with_name(checkpoint_stem))
    trainer.test(model, gazebase, ckpt_path=ckpt)
