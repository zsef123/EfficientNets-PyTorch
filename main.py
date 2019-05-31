import os
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from effnet import EfficientNet
from runner import Runner
from loader import get_loaders


from logger import Logger


def arg_parse():
    # projects description
    desc = "Pytorch EfficientNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory name to save the model')

    parser.add_argument('--root', type=str, default="/data2/imagenet",
                        help="The Directory of data path.")
    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbers | 0,1,2,3 | ")
    parser.add_argument('--num_workers', type=int, default="32",
                        help="Select CPU Number workers")

    parser.add_argument('--model', type=str, default='b0',
                        choices=["b0"],
                        help='The type of Efficient net.')

    parser.add_argument('--epoch', type=int, default=350, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='The size of batch')
    parser.add_argument('--test', action="store_true", help='Only Test')

    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help="Exponential Moving Average Term")

    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--dropconnect_rate', type=float, default=0.2)

    parser.add_argument('--optim', type=str, default='rmsprop', choices=["rmsprop"])
    parser.add_argument('--lr',    type=float, default=0.016, help="Base learning rate when train batch size is 256.")
    # Adam Optimizer
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--eps',      type=float, default=0.001)
    parser.add_argument('--decay',    type=float, default=1e-5)

    parser.add_argument('--scheduler', type=str, default='exp', choices=["exp", "cosine", "none"],
                        help="Learning rate scheduler type")

    return parser.parse_args()


def get_model(arg, classes=1000):
    if arg.model == "b0":
        return EfficientNet(1, 1, num_classes=classes)


def get_scheduler(optim, sche_type, step_size, t_max):
    if sche_type == "exp":
        return StepLR(optim, step_size, 0.97)
    elif sche_type == "cosine":
        return CosineAnnealingLR(optim, t_max)
    else:
        return None


if __name__ == "__main__":
    arg = arg_parse()

    arg.save_dir = "%s/outs/%s" % (os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    logger = Logger(arg.save_dir)
    logger.will_write(str(arg) + "\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    train_loader, val_loader = get_loaders(arg.root, arg.batch_size, 224, arg.num_workers)

    net = get_model(arg, classes=1000)
    net = nn.DataParallel(net).to(torch_device)
    loss = nn.CrossEntropyLoss()

    scaled_lr = arg.lr * arg.batch_size / 256
    optim = {
        # "adam" : lambda : torch.optim.Adam(net.parameters(), lr=arg.lr, betas=arg.beta, weight_decay=arg.decay),
        "rmsprop" : lambda : torch.optim.RMSprop(net.parameters(), lr=scaled_lr, momentum=arg.momentum, eps=arg.eps, weight_decay=arg.decay)
    }[arg.optim]()

    scheduler = get_scheduler(optim, arg.scheduler, int(2.4 * len(train_loader)), arg.epoch * len(train_loader))

    model = Runner(arg, net, optim, torch_device, loss, logger, scheduler)
    if arg.test is False:
        model.train(train_loader, val_loader)
    model.test(train_loader, val_loader)
