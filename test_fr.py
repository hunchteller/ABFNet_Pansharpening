import os
import argparse
import json
import torch
from dataloaders.full_res_dataset import DatasetFromFolder
import torch.utils.data as data
import torch.nn as nn
from utils.full_res_metrics_torch import no_ref_evaluate
import importlib
from models.abfnet import ABFNet
from concurrent.futures import ThreadPoolExecutor
import numpy as np
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
    torch.manual_seed(7)

    # Setting number of GPUS available for training.
    num_gpus = torch.cuda.device_count()

    # Selecting the model.
    batch_num = 128

    band = config['dataset']['spectral_bands']
    model = ABFNet(band=band)

    PATH = "./" + config["experim_name"]

    checkpoint = torch.load(PATH + "/" + "best_model.pth")
    model.load_state_dict(checkpoint, strict=True)

    print(f'\n{model}\n')


    # Sending model to GPU  device.
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).cuda()
    else:
        print("Single Cuda Node is avaiable")
        model.cuda()

    # Setting up training and testing dataloaderes.
    # print("Training with dataset => {}".format(config["train_dataset"]))

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
    test_set = DatasetFromFolder(config["dataset"]["full_res_dir"], is_train=False)

    test_loader = data.DataLoader(
        test_set,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    # Initialization of hyperparameters.

    # Resume...


    model.eval()
    criterion = nn.MSELoss()

    test_loss = 0.0
    D_lambda, D_s, QNR = 0, 0, 0
    val_outputs = {}
    model.eval()
    pred_dic = {}
    from tqdm import tqdm
    with torch.no_grad():
        for i, data in (enumerate(tqdm(test_loader), 0)):
            MS_image, PAN_image = data

            # Generating small patches
            # if config["trainer"]["is_small_patch_train"]:
            #     MS_image, unfold_shape = make_patches(MS_image, patch_size=config["trainer"]["patch_size"])
            #     PAN_image, _ = make_patches(PAN_image, patch_size=config["trainer"]["patch_size"])
            #     reference, _ = make_patches(reference, patch_size=config["trainer"]["patch_size"])

            # Inputs and references...
            MS_image = MS_image.float().cuda()  # .flatten(0, 1)
            PAN_image = PAN_image.float().cuda()  # .flatten(0, 1)
            # reference = reference.float().cuda()

            # if config[config['train_dataset']]['Crop_ratio'] > 1:
            #     MS_image = MS_image.flatten(0, 1)
            #     PAN_image = PAN_image.flatten(0, 1)

            # Taking model output
            outputs = model(MS_image, PAN_image)
            # outputs = out["pred"]
            # import ipdb; ipdb.set_trace()


            result = no_ref_evaluate(outputs, MS_image, PAN_image)
            D_lambda += result[0].sum()
            D_s += result[1].sum()
            QNR += result[2].sum()
            
            
            # outputs = rearrange(outputs, '(b h1 w1) c h w -> b c (h1 h) (w1 w)',
            #                     h1=config[config['train_dataset']]['Crop_ratio'],
            #                     w1=config[config['train_dataset']]['Crop_ratio'])

            # Computing validation loss
            # outputs = outputs.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            # MS_image = MS_image.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            # PAN_image = PAN_image.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            # for pr, ms, pan in zip(outputs, MS_image, PAN_image):
            #     m_dict = no_ref_evaluate(pr, pan, ms)
            #     D_lambda += m_dict['D_lambda']
            #     D_s += m_dict['D_s']
            #     QNR += m_dict["QNR"]
            ### Computing performance metrics ###

            # outputs = outputs.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            # MS_image = MS_image.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            # PAN_image = PAN_image.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            # futures_list = []
            # with ThreadPoolExecutor(max_workers=8) as pool:
            #     for pr, ms, pan in zip(outputs, MS_image, PAN_image):
            #         future = pool.submit(no_ref_evaluate, pr, pan, ms)
            #         futures_list.append(future)
            # result_list = [future.result() for future in futures_list]
            # # import ipdb; ipdb.set_trace()
            # result = torch.tensor(result_list)
            # print(result.shape)

            # D_lambda += result[:, 0].sum()
            # D_s += result[:, 1].sum()
            # QNR += result[:, 2].sum()


    # Taking average of performance metrics over test set
    D_lambda /= len(test_set)
    D_s /= len(test_set)
    QNR /= len(test_set)

    test_loss = 0

    print(
        f'epoch, loss:{test_loss:.4f}, D_lambda:{D_lambda:.4f}, D_s:{D_s:.4f}, QNR:{QNR:.4f}')
