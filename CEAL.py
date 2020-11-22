###################
# Pytorch neural network architecture
# with active learning
# Author: Martin Lund Haug
# email: martinlhaug@gmail.com
#
# Date created: 18 november 2020
# 
# Project: AILARON
#
#
# Copyright: @NTNU 2020
####################
import os
import torch
import pandas as pd
from skimage import io, transform
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, utils
import torchvision.models as models
import torch.nn.functional as F
import cv2
import csv
import time
from tqdm import tqdm
from metrics import METRIX
import torch
from torch import nn
from torch import optim
from dataloader import PlanktonDataSet, Resize, Normalize, ToTensor, Convert2RGB
from nn_models import AlexNet, ResNet152
from samples_selection import get_uncertain_samples
from criteria import least_confidence
import sys
import argparse
from torch.utils.data.sampler import SubsetRandomSampler


def load_data_pool(data_dir, header_file, filename, log_file, file_ending):
    
    fh = open(log_file, 'a+')
    fh.write('\n***** Loading dataset *****\n')

    composed = transforms.Compose([Convert2RGB(), Resize(64), Normalize(), ToTensor()])
        
    try:
        dataset = PlanktonDataSet(data_dir=data_dir, header_file=header_file, csv_file=filename, file_ending=file_ending,
                                    transform=composed)
    except Exception as e:
        fh.write('Could not load dataset, error msg: {}\n'.format(str(e)))
        print("Could not load dataset, error msg: ", str(e))
    fh.write('Len dataset: {}\n'.format(len(dataset)))
    fh.close()

    return dataset


def load_model(model_name, num_classes, log_file, size, device, num_channels):
    if model_name == "alexnet":
        net = AlexNet(num_classes, device)
    if model_name == "resnet152":
        net = ResNet152(num_classes, num_channels, device)
    return net.model    


def train(model, device, labeled_loader, optimizer, criterion):
    train_loss = 0
    model.train()
    model.to(device)
    for idx_batch, sample in tqdm(enumerate(labeled_loader),
                                total=len(labeled_loader),
                                leave=False):
        t0 = time.time()
        input, target = sample['image'], sample['label']
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(input.float())
        loss = criterion(outputs, target.squeeze(1).long())
        
        # Calculate gradients (backpropagation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        
        # Add loss to the training set running loss
        train_loss += loss.item() * input.size(0)
        t1 = time.time()
    return train_loss



def predict(model, device, unlabeled_loader, num_classes):

    model.eval()
    model.to(device)
    predict_results = np.empty(shape=(0, num_classes))
    with torch.no_grad(): # Check out what this does
        for sample in unlabeled_loader:
            data = sample['image']
            data = data.to(device)
            outputs = model(data.float())
            outputs = F.softmax(outputs)
            predict_results = np.concatenate(
                            (predict_results, outputs.cpu().numpy()))
    return predict_results


def test(model, device, criterion, test_loader, log_file):
    
    model.eval()
    model.to(device)
    test_loss = 0
    total = 0
    correct = 0
    accuracy = 0
    balanced_accuracy = 0

    for sample in test_loader:
        data, target = sample['image'], sample['label']
        data, target = data.to(device), target.to(device)
        outputs = model(data.float())
        loss = criterion(outputs, target.squeeze(1).long())
        test_loss += loss.item() * data.size(0)
        
        acc, bacc, precision, recall, f1_score, rep = \
            METRIX(target.squeeze(1).long(), outputs)
        accuracy += acc
        balanced_accuracy += bacc
        
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    return accuracy, balanced_accuracy
        
def run(device, log_file, epochs, batch_size,
        dataset, num_iter, start_lr, weight_decay, num_classes, criteria, k_samples, model_name, size, num_channels):


    fh = open(log_file, 'a+')
    fh.write('\n**** New CEAL **** \n')
    fh.write('INFO: Running on: {}, model name: {}, classes: {}, epochs: {}\n'
            'k: {}, criteria: {}, num iterations: {}, batch size: {}\n'.format(device, model_name, num_classes, epochs, k_samples, criteria, num_iter, batch_size))
    fh.close()
    
    
    criterion = nn.CrossEntropyLoss()
    iteration = 1
    # KFold validation
    kf = KFold(n_splits=5, random_state=None, shuffle=True)

    for train_index, test_index in kf.split(dataset):
        
        fh = open(log_file, 'a+')

        net = load_model(model_name, num_classes, log_file, size, device, num_channels)
        optimizer = optim.Adam(net.parameters(), lr=start_lr, weight_decay=weight_decay)
        net = net.float() 

        train_set = Subset(dataset, train_index)
        test_set = Subset(dataset, test_index)

        #train_set, test_set = dataset[train_index], dataset[test_index]
        fh.write('\nSplit up data, cross validation number: {}\n'.format(iteration))
        fh.write('len(train): {}, len(test): {}\n'.format(len(train_set), len(test_set)))
        # Define test data
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        iteration += 1
        # Define labeled and unlabeled data sets
        split = int(len(train_set) * 0.10) # 10% initial labeled data 
        indices = list(range(len(train_set)))

        unlabeled_indices, labeled_indices = indices[split:], indices[:split]
        #unlabeled, labeled = random_split(train_set,[len(train_set) - split_size,split_size])
        unlabeled_sampler = SubsetRandomSampler(unlabeled_indices)
        labeled_sampler = SubsetRandomSampler(labeled_indices)

        unlabeled_loader = DataLoader(train_set, batch_size=batch_size,
                                       sampler=unlabeled_sampler, shuffle=False)
        labeled_loader = DataLoader(train_set, batch_size=batch_size,
                                        sampler=labeled_sampler, shuffle=False)
        
        fh.write('len(labeled): {}\t len(unlabeled): {}\n'.
                format(len(labeled_loader.sampler.indices),len(unlabeled_loader.sampler.indices)))
        fh.close()

        flag = 0
        k = k_samples
        acc_list = []
        balacc_list = []

        for iter in range(num_iter):
            fh = open(log_file, 'a+')
            fh.write('\n** Active learning iteration: {} / {} **\n'.format(iter, num_iter))

            # ---------- Train model ----------- #
            fh.write('***** TRAIN *****\n')
            for epoch in range(1, epochs+1):
                t0 = time.time()
                train_loss = \
                    train(net, device, labeled_loader, optimizer, criterion)
                fh.write('\nTotal training time {:.2f} seconds\n'.format(time.time() - t0))
                train_loss = train_loss / len(labeled_loader.dataset)
                fh.write('Epoch:\t{}\tTraining Loss:\t{:.4f}\n'.format(epoch,train_loss))


            # ------------ Test model ------------- #
            fh.write('******* TEST *******\n')
            t0 = time.time()
            test_acc, test_balacc = test(net, device, criterion, test_loader, log_file)
            t1 = time.time()
            fh.write('Testing time\t{:.3f} seconds\n'.format(t1-t0))
            fh.write('Test acc:\t{:.3f}%\t'
                     'Test balacc:\t{:.3f}%\t'
                     'Fraction data: {:.3f}%\n'.format(test_acc*100/len(test_loader), test_balacc*100/len(test_loader),
                            100*len(labeled_loader.sampler.indices)/(len(labeled_loader.sampler.indices)+len(unlabeled_loader.sampler.indices))))

            if flag == 1:
                break

            # ---------- Active learning ----- #
            fh.write('***** ACTIVE LEARNING *****\n')
            pred_prob = predict(net, device, unlabeled_loader, num_classes)
            
            if len(pred_prob) < k:
                k = len(pred_prob)
                flag = 1

            # get k uncertain samples
            uncert_samp_idx = get_uncertain_samples(pred_prob=pred_prob, k=k,
                                                        criteria=criteria)
            # get original indices
            uncert_samp_idx = [unlabeled_loader.sampler.indices[idx] for idx in uncert_samp_idx]

            # add uncertain samples to labeled dataset
            labeled_loader.sampler.indices.extend(uncert_samp_idx)
            
            # remove the uncertain samples from the unlabeled pool
            for val in uncert_samp_idx:
                unlabeled_loader.sampler.indices.remove(val)

            fh.write('Update size of labeled and unlabeled dataset by adding {} uncertain samples\n'
                    'updated len(labeled): {}\t updated len(unlabeled): {}\n'.
                    format(len(uncert_samp_idx),len(labeled_loader.sampler.indices),len(unlabeled_loader.sampler.indices)))


            fh.close()
          

def benchmark(device, log_file, bench_epochs, batch_size, dataset, start_lr, weight_decay, num_classes, model_name, size, num_channels):

    fh = open(log_file, 'a+')
    fh.write('\n**** New BENCHMARK **** \n')
    fh.write('INFO: Running on: {}, model name: {}, classes: {}, epochs: {}, batch size: {}\n'.format(device, model_name, num_classes, bench_epochs, batch_size))
    fh.close()

    criterion = nn.CrossEntropyLoss()
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    iteration = 1

    for train_index, test_index in kf.split(dataset):

        fh = open(log_file, 'a+')    
        net = load_model(model_name, num_classes, log_file, size, device, num_channels)
        optimizer = optim.Adam(net.parameters(), lr=start_lr, weight_decay=weight_decay)
        net = net.float()

        train_set = Subset(dataset, train_index)
        test_set = Subset(dataset, test_index)

        fh.write('\nSplit up data, cross validation number: {}\n'.format(iteration))
        fh.write('len(train): {}, len(test): {}\n'.format(len(train_set), len(test_set)))

        iteration += 1
        # Define test data
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        fh.close()
        fh = open(log_file, 'a+')    
        # ---------- Train model ----------- #
        fh.write('***** TRAIN *****\n')
        fh.close()
        for epoch in range(1, bench_epochs+1):
            fh = open(log_file, 'a+')    
            t0 = time.time()
            train_loss = \
                train(net, device, train_loader, optimizer, criterion)
            fh.write('\nTotal training time {:.2f} seconds\n'.format(time.time() - t0))
            train_loss = train_loss / len(train_loader.dataset)
            fh.write('Epoch:\t{}\tTraining Loss:\t{:.4f}\n'.format(epoch,train_loss))   
            fh.close()    

        # ------------ Test model ------------- #
        fh = open(log_file, 'a+')    
        fh.write('******* TEST *******\n')
        t0 = time.time()
        test_acc, test_balacc = test(net, device, criterion, test_loader, log_file)
        t1 = time.time()
        fh.write('Testing time\t{:.3f} seconds\n'.format(t1-t0))
        fh.write('Test acc:\t{:.3f}%\t'
                 'Test balacc:\t{:.3f}%\t\n'.format(test_acc*100/len(test_loader), test_balacc*100/len(test_loader)))
        fh.close()



if __name__ == "__main__":

    weight_decay = 0.0001
    start_lr = 0.001
    # data_dir = "/Users/martin.lund.haug/Documents/Prosjektoppgave/Datasets/plankton_new_data/Dataset_BeringSea/train/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    data_dir = sys.argv[1]+"/"    
    header_file = data_dir + "header.tfl.txt"
    filename = data_dir + "image_set.data"
    log_file = data_dir + "_run.log"
    file_ending = ".jpg"
    model_name = sys.argv[2]
    num_classes = int(sys.argv[3]) #7 # DYNAMIC
    size = 64
    num_channels = 3
    epochs = 10  # Add break when training loss stops decreasing 
    bench_epochs = 20
    batch_size = 16
    num_iter = 40
    criterias = ["rd", "en", "ms", "lc"]
    k_samples = 700



    dataset = load_data_pool(data_dir, header_file, filename, log_file, file_ending)
    #run(device, log_file, epochs, batch_size, dataset, num_iter, start_lr, weight_decay, num_classes, criteria, k, model_name, size, num_channels)
    #benchmark(device, log_file, bench_epochs, batch_size, dataset, start_lr, weight_decay, num_classes, model_name, size, num_channels)

    for criteria in criterias:
        run(device, log_file, epochs, batch_size, dataset, num_iter, start_lr, weight_decay, num_classes, criteria, k_samples, model_name, size, num_channels)
