
from typing import Optional, Callable

from torchvision.models import alexnet, resnet152, resnet18, resnet34
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.optim as Optimizer
import logging
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
from torchvision import models, transforms

class ResNet152(object):

    def __init__(self, n_classes, n_channels, device):

        self.n_classes = n_classes
        self.model = resnet152(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))
        
    def __change_last_layer(self) -> None:
        self.model.fc = torch.nn.Linear(2048, self.n_classes)
        
class ResNet18(object):

    def __init__(self, n_classes, n_channels, device):
        self.n_classes = n_classes
        self.model = resnet18(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))

    def __change_last_layer(self) -> None:
        self.model.fc = torch.nn.Linear(512, self.n_classes)

class ResNet34(object):

    def __init__(self, n_classes, n_channels, device):
        self.n_classes = n_classes
        self.model = resnet34(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))

    def __change_last_layer(self) -> None:
        self.model.fc = torch.nn.Linear(512, self.n_classes)


class AlexNet(object):


    def __init__(self, n_classes, device):

        self.n_classes = n_classes
        self.model = alexnet(pretrained=True, progress=True)
        self.device = device
        print("The code is running on {}".format(self.device))
        self.__change_last_layer()

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

    model = ResNet18(7, 3, "gpu")
    print(model.model)

    model_weights = []
    conv_layers = []
    model_children = list(model.model.children())

    # counter to keep count of the conv layers
    counter = 0 
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    # visualize the first conv layer filters
    plt.figure(figsize=(10, 7))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        #plt.savefig('../outputs/filter.png')
    plt.show()


    # read and visualize an image
    path = "/Users/martin.lund.haug/Documents/Prosjektoppgave/Datasets/plankton_new_data/Dataset_BeringSea/train/Copepoda/copepoda_490.bmp"
    img = cv.imread(path)
    print(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    print(img.size())
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))

    # make a copy of the `results`
    outputs = results

    
    # visualize 64 features from each layer 
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64: # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"../conv_layers/layer_{num_layer}.eps")
        #plt.show()
        #plt.close()
        