
from typing import Optional, Callable

from torchvision.models import alexnet, resnet152
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.optim as Optimizer
import logging


class ResNet152(object):

    def __init__(self, n_classes, device):

        self.n_classes = n_classes
        self.model = resnet152(pretrained=True, progress=True)
        self.__change_last_layer()
        if device is None:
            self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu")

    def __change_last_layer():
        pass
        



class AlexNet(object):


    def __init__(self, n_classes, device):

        self.n_classes = n_classes
        self.model = alexnet(pretrained=True, progress=True)

        self.__change_last_layer()
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

    def __freeze_all_layers(self) -> None:
        """
        freeze all layers in alexnet
        Returns
        -------
        None
        """

        for param in self.model.parameters():
            param.requires_grad = False

    def __change_last_layer(self) -> None:
        """
        change last layer to accept n_classes instead of 1000 classes
        Returns
        -------
        None
        """
        self.model.classifier[6] = nn.Linear(4096, self.n_classes)

    def __add_softmax_layer(self) -> None:
        """
        Add softmax layer to alexnet model
        Returns
        -------

        """
        # add softmax layer
        self.model = nn.Sequential(self.model, nn.LogSoftmax(dim=1))


if __name__ == "__main__":

    model = AlexNet(7, "gpu")
    print(model.model)