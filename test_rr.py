import os
import argparse
import json
import torch
import numpy as np
from torch.nn.functional import threshold, unfold
from dataloaders.dataset import DatasetFromFolder
from utils.logger import Logger
import torch.utils.data as data
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.abfnet import ABFNet
from utils.batch_metrics import *
import shutil
import torchvision
from torch.distributions.uniform import Uniform
import sys
from einops import rearrange
from utils.utility import LossMeter


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':

    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_QB.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    # Loading the config file
    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True

    # Set seeds.
    torch.manual_seed(7)

    # Setting number of GPUS available for training.
    num_gpus = torch.cuda.device_count()


    # Selecting the model.
    band = config['dataset']['spectral_bands']
    model = ABFNet(band=band)
    print(f'\n{model}\n')

    # Sending model to GPU  device.
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).cuda()
    else:
        print("Single Cuda Node is avaiable")
        model.cuda()


    # train_set = DatasetFromFolder(config["dataset"]["data_dir"], is_train=True)
    #
    # train_loader = data.DataLoader(
    #     train_set,
    #     batch_size=config["train_batch_size"],
    #     num_workers=config["num_workers"],
    #     shuffle=True,
    #     pin_memory=False,
    #     drop_last=True,
    # )

    test_set = DatasetFromFolder(config["dataset"]["data_dir"], is_train=False)

    test_loader = data.DataLoader(
        test_set,
        batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=False,
    )

    # Initialization of hyperparameters.

    # Resume...
    PATH = "./" + config["experim_name"]
    checkpoint = torch.load(PATH + "/" + "best_model.pth")
    model.load_state_dict(checkpoint, strict=True)

    model.eval()
    criterion = nn.MSELoss()

    test_loss = 0.0
    cc = 0.0
    sam = 0.0
    rmse = 0.0
    ergas = 0.0
    psnr = 0.0
    val_outputs = {}
    model.eval()
    pred_dic = {}
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            MS_image, PAN_image, reference = data

            # Generating small patches
            # if config["trainer"]["is_small_patch_train"]:
            #     MS_image, unfold_shape = make_patches(MS_image, patch_size=config["trainer"]["patch_size"])
            #     PAN_image, _ = make_patches(PAN_image, patch_size=config["trainer"]["patch_size"])
            #     reference, _ = make_patches(reference, patch_size=config["trainer"]["patch_size"])

            # Inputs and references...
            MS_image = MS_image.float().cuda()  # .flatten(0, 1)
            PAN_image = PAN_image.float().cuda()  # .flatten(0, 1)
            reference = reference.float().cuda()

            # if config[config['train_dataset']]['Crop_ratio'] > 1:
            #     MS_image = MS_image.flatten(0, 1)
            #     PAN_image = PAN_image.flatten(0, 1)

            # Taking model output
            outputs = model(MS_image, PAN_image)
            # outputs = out["pred"]

            # outputs = rearrange(outputs, '(b h1 w1) c h w -> b c (h1 h) (w1 w)',
            #                     h1=config[config['train_dataset']]['Crop_ratio'],
            #                     w1=config[config['train_dataset']]['Crop_ratio'])

            # Computing validation loss
            loss = criterion(outputs, reference)

            ### Computing performance metrics ###
            # Cross-correlation
            cc += batch_cross_correlation(outputs, reference).cpu()
            # SAM
            sam += batch_SAM(outputs, reference).cpu()
            # RMSE
            rmse += batch_RMSE(outputs / torch.max(reference), reference / torch.max(reference)).cpu()
            # ERGAS
            beta = torch.tensor(
                config["dataset"]["HR_size"] / config["dataset"]["LR_size"]).cuda()
            ergas += batch_ERGAS(outputs, reference, beta).cpu()
            # PSNR
            psnr += batch_PSNR(outputs, reference, config["dataset"]["Bit_depth"]).cpu()

    # Taking average of performance metrics over test set
    cc /= len(test_set)
    sam /= len(test_set)
    rmse /= len(test_set)
    ergas /= len(test_set)
    psnr /= len(test_set)
    test_loss = loss

    print(
        f'epoch:test, loss:{test_loss:.4f}, cc:{cc:.4f}, sam:{sam:.4f}, rmse:{rmse:.4f}, ergas:{ergas:.4f}, psnr:{psnr:.4f}')
