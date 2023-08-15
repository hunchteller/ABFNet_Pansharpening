import os
import argparse
import json
import torch
import numpy as np
from torch.nn.functional import threshold, unfold
from dataloaders.dataset import DatasetFromFolder

import torch.utils.data as data
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.abfnet import ABFNet
from utils.batch_metrics import *
import shutil
import sys
from einops import rearrange


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
    import random
    torch.manual_seed(7)
    np.random.seed(7)
    torch.cuda.manual_seed(7)
    random.seed(7)

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

    # Setting up training and testing dataloaderes.

    train_set = DatasetFromFolder(config["dataset"]["data_dir"], is_train=True)

    train_loader = data.DataLoader(
        train_set,
        batch_size=config["train_batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    test_set = DatasetFromFolder(config["dataset"]["data_dir"], is_train=False)

    test_loader = data.DataLoader(
        test_set,
        batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=False,
    )

    # Initialization of hyperparameters.
    start_epoch = 1
    total_epochs = config["trainer"]["total_epochs"] + 1

    # Setting up optimizer.
    if config["optimizer"]["type"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["optimizer"]["args"]["lr"],
            momentum=config["optimizer"]["args"]["momentum"],
            weight_decay=config["optimizer"]["args"]["weight_decay"]
        )
    elif config["optimizer"]["type"] == "ADAM":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["optimizer"]["args"]["lr"],
            weight_decay=config["optimizer"]["args"]["weight_decay"]
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["optimizer"]["args"]["lr"],
            # weight_decay=config["optimizer"]["args"]["weight_decay"]
        )

    # Learning rate sheduler.
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config["optimizer"]["step_size"],
                                          gamma=config["optimizer"]["gamma"])
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["optimizer"]["multi_step_size"],
    #                                            gamma=config["optimizer"]["gamma"])

    # Resume...
    if args.resume is not None:
        print("Loading from existing FCN and copying weights to continue....")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint, strict=False)
    else:
        # initialize_weights(model)
        initialize_weights_new(model)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()


    # Training epoch.
    def train(epoch):
        train_loss = 0.0
        model.train()
        optimizer.zero_grad()
        psnr = 0
        for i, data in enumerate(train_loader, 0):
            # Reading data.
            MS_image, PAN_image, reference = data

            # Taking model outputs ...
            MS_image = MS_image.float().cuda()
            PAN_image = PAN_image.float().cuda()
            reference = reference.float().cuda()

            outputs = model(MS_image, PAN_image)
            # outputs = out["pred"]

            loss = criterion(outputs, reference)
            loss.backward()

            if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                psnr += batch_PSNR(outputs, reference, config["dataset"]["Bit_depth"]).cpu() / outputs.size(0)

        writer.add_scalar('Loss/train', loss, epoch)
        print(f'epoch:{epoch}, loss:{loss.item():.4f}, psnr:{psnr.item() / len(train_loader):.4f}')


    # Testing epoch.
    def test(epoch):
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
                #     MS_image = rearrange(MS_image, 'b c (h1 h) (w1 w)-> (b h1 w1) c h w', h1=crop_ratio, w1=crop_ratio)
                #     PAN_image = rearrange(PAN_image, 'b c (h1 h) (w1 w)-> (b h1 w1) c h w', h1=crop_ratio, w1=crop_ratio)


                # Taking model output
                outputs = model(MS_image, PAN_image)

                # outputs = out["pred"]

                # if crop_ratio > 1:
                #     outputs = rearrange(outputs, '(b h1 w1) c h w -> b c (h1 h) (w1 w)',
                #                         h1=config[config['train_dataset']]['Crop_ratio'],
                #                         w1=config[config['train_dataset']]['Crop_ratio'])

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

        # Writing test results to tensorboard
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Test_Metrics/CC', cc, epoch)
        writer.add_scalar('Test_Metrics/SAM', sam, epoch)
        writer.add_scalar('Test_Metrics/RMSE', rmse, epoch)
        writer.add_scalar('Test_Metrics/ERGAS', ergas, epoch)
        writer.add_scalar('Test_Metrics/PSNR', psnr, epoch)
        print(
            f'epoch:{epoch}, loss:{test_loss:.4f}, cc:{cc:.4f}, sam:{sam:.4f}, rmse:{rmse:.4f}, ergas:{ergas:.4f}, psnr:{psnr:.4f}')

        # Return Outputs
        metrics = {"loss": float(test_loss),
                   "cc": float(cc),
                   "sam": float(sam),
                   "rmse": float(rmse),
                   "ergas": float(ergas),
                   "psnr": float(psnr)}
        return pred_dic, metrics


    # Setting up tensorboard and copy .json file to save directory.
    PATH = "./" + config["experim_name"]
    ensure_dir(PATH + "/")
    writer = SummaryWriter(log_dir=PATH)
    shutil.copy2(args.config, PATH)

    # Print model to text file
    original_stdout = sys.stdout
    with open(PATH + "/" + "model_summary.txt", 'w+') as f:
        sys.stdout = f
        print(f'\n{model}\n')
        sys.stdout = original_stdout

    # Main loop.
    best_psnr = 0.0
    for epoch in range(start_epoch, total_epochs):
        scheduler.step(epoch)
        print("\nTraining Epoch: %d" % epoch)
        train(epoch)

        if epoch % config["trainer"]["test_freq"] == 0:
            print("\nTesting Epoch: %d" % epoch)
            pred_dic, metrics = test(epoch)

            # Saving the best model
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]

                # Saving best performance metrics
                torch.save(model.state_dict(), PATH + "/" + "best_model.pth")
                with open(PATH + "/" + "best_metrics.json", "w+") as outfile:
                    json.dump(metrics, outfile)

                # Saving best prediction
                # savemat(PATH + "/" + "final_prediction.mat", pred_dic)
