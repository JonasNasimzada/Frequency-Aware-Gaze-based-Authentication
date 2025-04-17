# Frequency-Aware Gaze-based Authentication

Code is based on the "[Eye Know You Too](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/61ZGZN)" paper

Implemented by: \
Jonas Nasimzada \
Álvaro González Tabernero \
Louis Beaudoing \
Pranav Abraham Mathews 

## Overview
This is the source code for our eye movement biometrics model, Eye Know You Too.

## Citation of the "Eye Know You Too" paper
D. Lohr and O. V. Komogortsev, "Eye Know You Too: Toward Viable End-to-End Eye
Movement Biometrics for User Authentication," in *IEEE Transactions on
Information Forensics and Security*, 2022, doi: 10.1109/TIFS.2022.3201369.

## License
This work is licensed under a "Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International License"
(https://creativecommons.org/licenses/by-nc-sa/4.0/).


## Setting up a compatible [Anaconda](https://www.anaconda.com/) environment
```bash
$ conda create -n ekyt-release python==3.7.11
$ conda activate ekyt-release

# PyTorch (different setups may require a different version of cudatoolkit)
$ conda install -c pytorch -c conda-forge pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit==11.3.1

# PyTorch Metric Learning (for multi-similarity loss and, optionally, MAP@R)
$ conda install -c metric-learning -c pytorch pytorch-metric-learning==0.9.99

# PyTorch Lightning and other necessary packages
$ conda install -c conda-forge pytorch-lightning==1.5.0 pandas==1.3.4 tensorboard==2.6.0 scikit-learn==1.0.1 numpy==1.21.2 scipy==1.7.1 tqdm==4.62.3

# (optional) For computing MAP@R
$ conda install -c conda-forge faiss-gpu==1.7.1

# (optional) For plotting figures
$ conda install -c conda-forge matplotlib==3.4.3 umap-learn==0.5.1

# (optional) For formatting source code
$ conda install -c conda-forge black flake8
```

## Training a model
If you are only interested in using our provided pre-trained models, you can
skip this section.
```bash
# See all possible command-line arguments for training and testing
$ python train_and_test.py --help

# Train a model with default settings.  The first time this script is
# run for a given `--ds` and/or with the flag `--degrade_precision`, the
# dataset will need to be prepared.
$ python train_and_test.py --mode=train

# (optional) Track the model's progress with Tensorboard
$ tensorboard --logdir=lightning_logs

# Train a full ensemble to enable evaluation
$ python train_and_test.py --mode=train --fold=0
$ python train_and_test.py --mode=train --fold=1
$ python train_and_test.py --mode=train --fold=2
$ python train_and_test.py --mode=train --fold=3
```

## Testing a model (i.e., computing embeddings for later evaluation)
```bash
# See all possible command-line arguments for training and testing
$ python train_and_test.py --help

# Test a model that was trained with default settings.  The first time
# this script is run for a given `--ds` and/or with the flag
# `--degrade_precision`, the dataset will need to be prepared.
$ python train_and_test.py --mode=test

# Test a full ensemble to enable evaluation
$ python train_and_test.py --mode=test --fold=0
$ python train_and_test.py --mode=test --fold=1
$ python train_and_test.py --mode=test --fold=2
$ python train_and_test.py --mode=test --fold=3

# If you have less than 16 GB of VRAM and are working with our pre-
# trained models on 1000 Hz data, you might need to reduce the batch
# size at test time
$ python train_and_test.py --mode=test --batch_size_for_testing=64
```

## Evaluating a model
```bash
# See all possible command-line arguments for evaluation.  Note that we
# evaluate on one task (--task), one round (--round), and one duration
# (--n_seq) at a time.
$ python evaluate_[combination|recording_duration].py --help

# Evaluate the ensemble model that was trained with default settings.
# We evaluate under the primary evaluation scenario by default (the
# first 5 seconds during R1 TEX).
$ python evaluate_[combination|recording_duration].py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal

```
