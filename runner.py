import os
import time
from glob import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Runner():
    def __init__(self, arg, net, optim, torch_device, loss, logger):
        self.arg = arg
        self.save_dir = arg.save_dir

        self.logger = logger

        self.torch_device = torch_device

        self.net = net
        self.loss = loss
        self.optim = optim

        self.start_epoch = 0
        self.best_metric = -1

        self.load()

    def save(self, epoch, filename="train"):
        """Save current epoch model

        Save Elements:
            model_type : arg.model
            start_epoch : current epoch
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score

        Parameters:
            epoch : current epoch
            filename : model save file name
        """

        torch.save({"model_type": self.arg.model,
                    "start_epoch": epoch + 1,
                    "network": self.net.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("Model saved %d epoch" % (epoch))

    def load(self, filename=""):
        """ Model load. same with save"""
        if filename == "":
            # load last epoch model
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[-1])

        file_path = self.save_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File" % (self.save_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s" %
                                 (ckpoint["model_type"]))

            self.net.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f" %
                  (ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def train(self, train_loader, val_loader=None):
        print("\nStart Train len :", len(train_loader.dataset))
        self.net.train()
        for epoch in range(self.start_epoch, self.arg.epoch):
            for i, (input_, target_) in enumerate(train_loader):
                target_ = target_.to(self.torch_device, non_blocking=True)
                out = self.net(input_)
                loss = self.loss(out, target_)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if (i % 50) == 0:
                    self.logger.log_write("train", epoch=epoch, loss=loss.item())
            if val_loader is not None:
                self.valid(epoch, val_loader)

    def _get_acc(self, loader):
        correct = 0
        for input_, target_ in loader:
            target_ = target_.to(self.torch_device, non_blocking=True)
            out = self.net(input_)
            out = F.softmax(out, dim=1)

            _, idx = out.max(dim=1)
            correct += (target_ == idx).sum().cpu().item()
        return correct / len(loader.dataset)

    def valid(self, epoch, val_loader):
        self.net.eval()
        with torch.no_grad():
            acc = self._get_acc(val_loader)
            self.logger.log_write("valid", epoch=epoch,
                                  acc=acc, test_acc=test_acc)

            if acc > self.best_metric and epoch > 50:
                self.best_metric = acc
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]" % (
                    epoch, acc, test_acc))

    def test(self, train_loader, val_loader):
        print("\n Start Test")
        self.load()
        self.net.eval()
        with torch.no_grad():
            train_acc = self._get_acc(train_loader)
            valid_acc = self._get_acc(val_loader)
            self.logger.log_write("test", fname="test", train_acc=train_acc, valid_acc=valid_acc)
        return train_acc, valid_acc
