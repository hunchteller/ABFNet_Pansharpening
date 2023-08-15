import torch
import numpy as np
import random
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def print_loss_dict(loss_dict):
    plist = []
    for k, v in loss_dict.items():
        if 'loss' in k:
            plist.append(f'{k}:{v:.4f}\t')
        else:
            plist.append(f'loss_{k}:{v:.4f}\t')
    return ''.join(plist)


def get_time(format='%Y-%m-%d %H:%M:%S'):
    return datetime.strftime(datetime.now(), format=format)


def init_logger(cfg):
    """
    initialization the logger part; tensorboard, logger, save_dir
    Args:
        cfg: configs

    Returns:
        writer, logger, save_dir
    """
    cfg.desc = cfg.desc

    os.makedirs(osp.join(cfg.dir, cfg.desc, 'tb'), exist_ok=True)
    os.makedirs(osp.join(cfg.dir, cfg.desc, 'ckpt'), exist_ok=True)

    writer = SummaryWriter(osp.join(cfg.dir, cfg.desc, 'tb'))
    logger = Logger(osp.join(cfg.dir, cfg.desc, f'{cfg.desc}_log'))
    ckpt_dir = osp.join(osp.join(cfg.dir, cfg.desc, 'ckpt'))
    return writer, logger, ckpt_dir


class LossMeter():
    def __init__(self):
        self.loss = 0
        self.num = 0

    def add(self, loss, num):
        self.loss += loss * num
        self.num += num

    def avg(self):
        return self.loss / self.num

    def reset(self):
        self.loss = 0
        self.num = 0


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def z_save_model(model, opt, sch, epoch, save_path):
    save_dict = {
        'model': model.state_dict(),
        'optim': opt.state_dict(),
        'sch': sch.state_dict(),
        'epoch': epoch}
    torch.save(save_dict, save_path)


def z_load_model(model, opt, sch, save_path):
    save_dict = torch.load(save_path)
    model.load_state_dict(save_dict['model'])
    opt.load_state_dict(save_dict['optim'])
    sch.load_state_dict(save_dict['sch'])
    return save_dict['epoch']


if __name__ == '__main__':
    print(get_time())
