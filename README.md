# HEANet

A transformer-based GNN model for learning gibbs energy from high-entropy alloys.

## Requirement

The important packages are presented as follows:

```
e3nn                  0.4.4
numpy                 1.22.4
pymatgen              2023.2.28
scipy                 1.8.1
timm                  0.4.12
torch                 1.10.2+cu111
torch-cluster         1.6.0
torch-geometric       2.2.0
torch-scatter         2.0.9
torch-sparse          0.6.13
torch-spline-conv     1.2.1
torchaudio            0.10.2 
torchmetrics          0.8.2
torchvision           0.11.3+cu111
tqdm                  4.65.0 
```

## Dataset

The dataset is undered `datasets/hea_4k/` and the DFT calcualted gibbs energy values are saved in `datasets/hea_4k/gibbs_eng.csv`. The `initial_p` includes primitive structures before DFT relax, and the `relaxed_p` includes primitive structures after DFT relax.

## Training

The input data will be divided into k-fold before training, so you can train according to the number of folds you want to run by setting `fold` args.
For example:

```
    sh scripts/train.sh
```

## Inference

Pretrained models are stored in `models/hea`. All customized training models will be saved in `${output_dir}`. 
You can use these saved `*.pt` files for inference with the following commands:

```
    sh scripts/test.sh
```
The results of inference will be saved in `predict.csv`.