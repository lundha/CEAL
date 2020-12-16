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
from torchvision.datasets import CIFAR10
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
from nn_models import AlexNet, ResNet152, ResNet18, ResNet34
from samples_selection import get_uncertain_samples, get_high_confidence_samples
from criteria import least_confidence
import sys
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from operator import add
from subset import my_subset

def load_data_pool(data_dir, header_file, filename, log_file, file_ending, num_classes):
    
    fh = open(log_file, 'a+')
    fh.write('\n***** Loading dataset *****\n')

    composed = transforms.Compose([Convert2RGB(), Resize(224), Normalize(), ToTensor()])
        
    try:
        dataset = PlanktonDataSet(data_dir=data_dir, header_file=header_file, csv_file=filename, file_ending=file_ending,
                                    transform=composed, num_classes=num_classes)
    except Exception as e:
        fh.write('Could not load dataset, error msg: {}\n'.format(str(e)))
        print("Could not load dataset, error msg: ", str(e))
    fh.write('Len dataset: {}\n'.format(len(dataset)))
    fh.close()

    return dataset

def load_cifar(log_file):

    fh = open(log_file, 'a+')
    fh.write('\n***** Loading dataset *****\n')

    composed = transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
    try:
        dataset = CIFAR10(root='/home/martlh/Documents/cifar10/cifar-10-batches-py', train=True, download=True, transform=composed)
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
    if model_name == "resnet18":
        net = ResNet18(num_classes, num_channels, device)
    if model_name == "resnet34":
        net = ResNet34(num_classes, num_channels, device)

    return net.model    


def train(model, device, labeled_loader, optimizer, criterion, use_cifar):
    train_loss = 0
    model.train()
    model.to(device)
    for idx_batch, sample in tqdm(enumerate(labeled_loader),
                                total=len(labeled_loader),
                                leave=False):
        t0 = time.time()
        if use_cifar == 1:
            input, target = sample[0], sample[1]
        else:
            input, target = sample['image'], sample['label']
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(input.float())
        if use_cifar == 1:
             loss = criterion(outputs, target.long())
        else:
            loss = criterion(outputs, target.squeeze(1).long())
        
        # Calculate gradients (backpropagation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        
        # Add loss to the training set running loss
        train_loss += loss.item() * input.size(0)
        t1 = time.time()
    return train_loss



def predict(model, device, unlabeled_loader, num_classes, use_cifar):

    model.eval()
    model.to(device)
    predict_results = np.empty(shape=(0, num_classes))
    with torch.no_grad(): 
        for sample in unlabeled_loader:
            if use_cifar == 1:
                data = sample[0]
            else:
                data = sample['image']
            data = data.to(device)
            outputs = model(data.float())
            outputs = F.softmax(outputs)
            predict_results = np.concatenate(
                            (predict_results, outputs.cpu().numpy()))
    return predict_results


def test(model, device, criterion, test_loader, log_file, use_cifar):
    
    model.eval()
    model.to(device)
    test_loss = 0
    total = 0
    correct = 0
    accuracy = 0
    balanced_accuracy = 0
    precision = 0
    ceal_acc = 0
  
    with torch.no_grad():
        for sample in test_loader:
            if use_cifar == 1:
                data, target = sample[0], sample[1]
            else:
                data, target = sample['image'], sample['label']
            data, target = data.to(device), target.to(device)
            outputs = model(data.float())
            if use_cifar == 1:
                loss = criterion(outputs, target.long())
            else:
                loss = criterion(outputs, target.squeeze(1).long())
            test_loss += loss.item() * data.size(0)
            
            if use_cifar == 1:
                acc, bacc, prec, recall, f1_score, rep = METRIX(target.long(), outputs)
            else:
                acc, bacc, prec, recall, f1_score, rep = METRIX(target.squeeze(1).long(), outputs)
            accuracy += acc
            balanced_accuracy += bacc
            precision += prec

    return accuracy, balanced_accuracy, precision


def run(device, log_file, epochs, batch_size,
        dataset, num_iter, start_lr, weight_decay, num_classes, criteria, k_samples, model_name, size, num_channels, delta_0, bool_ceal, use_cifar):
    t00 = time.time()

    fh = open(log_file, 'a+')
    fh.write('\n**** New CEAL **** \n')
    fh.write('INFO: Running on: {}, model name: {}, classes: {}, epochs: {}\n'
            'k: {}, criteria: {}, num iterations: {}, batch size: {}\n'.format(device, model_name, num_classes, epochs, k_samples, criteria, num_iter, batch_size))
    fh.close()
    tot_acc = [0]*200
    tot_balacc = [0]*200
    tot_uncert = [0]*200
    tot_precision = [0]*200
    tot_train_time = [0]*200
    tot_len_labeled_samples = [0]*200
    classCount = [0]*num_classes
    
    criterion = nn.CrossEntropyLoss()
    iteration = 1
    stop_flag = 0
    # KFold validation
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    
    dataset_size = len(dataset)
    for train_index, test_index in kf.split(dataset):
        
        if stop_flag == 1:
            break

        fh = open(log_file, 'a+')

        net = load_model(model_name, num_classes, log_file, size, device, num_channels)
        optimizer = optim.Adam(net.parameters(), lr=start_lr, weight_decay=weight_decay)
        net = net.float() 

       # train_set = Subset(dataset, train_index)
       # test_set = Subset(dataset, test_index)
        
        
        train_labels = []
        test_labels = []

        if use_cifar == 1:
            for idx, sample in enumerate(dataset):
                if idx in train_index:
                    train_labels.append(sample[1])
                else:
                    test_labels.append(sample[1])
        else:
            for idx, sample in enumerate(dataset):
                if idx in train_index:
                    train_labels.append(sample['label'])
                else:
                    test_labels.append(sample['label'])

        train_set = my_subset(dataset, train_index, train_labels)
        test_set  = my_subset(dataset, test_index, test_labels)
        

        train_set_size = len(train_set)
        #train_set, test_set = dataset[train_index], dataset[test_index]
        fh.write('\nSplit up data, cross validation number: {}\n'.format(iteration))
        fh.write('len(train): {}, len(test): {}\n'.format(len(train_set), len(test_set)))
        # Define test data
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        iteration += 1
        # Define labeled and unlabeled data sets
        split = int(len(train_set) * 0.010) # 10% initial labeled data 
        indices = list(range(len(train_set)))

        unlabeled_indices, labeled_indices = indices[split:], indices[:split]

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
        fraction = [0]
        acc_list = [0]
        balacc_list = [0]
        precision_list = [0]
        train_time = []
        hcs_idx = []
        uncert_prob_list = []
        len_labeled_samples = []
        fh = open(log_file, 'a+')

        for iter in range(num_iter):
            fh = open(log_file, 'a+')
            fh.write('\n** Active learning iteration: {} / {} **\n'.format(iter, num_iter))

            len_labeled_samples.append(len(labeled_loader.sampler.indices))

            # ---------- Train model ----------- #
            fh.write('***** TRAIN *****\n')
            t0_train = time.time()
            for epoch in range(1, epochs+1):
                t0 = time.time()
                train_loss = \
                    train(net, device, labeled_loader, optimizer, criterion, use_cifar)
                fh.write('\nTotal training time {:.2f} seconds\n'.format(time.time() - t0))
                train_loss = train_loss / len(labeled_loader.dataset)
                fh.write('Epoch:\t{}\tTraining Loss:\t{:.4f}\n'.format(epoch,train_loss))
            t1_train = time.time()
            train_time.append(t1_train-t0_train)

            if bool_ceal == True:
                # remove the high certain samples from the labeled pool after training
                for val in hcs_idx:
                    labeled_loader.sampler.indices.remove(val)
                fh.write('\n** Removed {} hcs from labeled samples\n'.format(len(hcs_idx)))


            # ------------ Test model ------------- #
            fh.write('******* TEST *******\n')
            t0 = time.time()
            test_acc, test_balacc, precision = test(net, device, criterion, test_loader, log_file, use_cifar)
            t1 = time.time()
            fh.write('Testing time\t{:.3f} seconds\n'.format(t1-t0))
            fh.write('Test acc:\t{:.3f}%\t'
                     'Test balacc:\t{:.3f}%\t'
                     'Test precision:\t{:.3f}%\t'
                     'Fraction data: {:.3f}%\n'.format(test_acc*100/len(test_loader), test_balacc*100/len(test_loader), precision*100/len(test_loader),
                            100*len(labeled_loader.sampler.indices)/train_set_size))

            fraction.append(100*len(labeled_loader.sampler.indices)/train_set_size)
            acc_list.append(test_acc*100/len(test_loader))
            balacc_list.append(test_balacc*100/len(test_loader))
            precision_list.append(precision*100/len(test_loader))

            if flag == 1:
                break

            # ---------- Active learning ----- #
            fh.write('***** ACTIVE LEARNING *****\n')
            pred_prob = predict(net, device, unlabeled_loader, num_classes, use_cifar)
            
            if len(pred_prob) < k:
                k = len(pred_prob)
                flag = 1

            # get k uncertain samples
            uncert_samp_idx, uncert_prob = get_uncertain_samples(pred_prob=pred_prob, k=k,
                                                        criteria=criteria)
            # get original indices
            uncert_samp_idx = [unlabeled_loader.sampler.indices[idx] for idx in uncert_samp_idx]
            
                        
            
            # add uncertain samples to labeled dataset
            labeled_loader.sampler.indices.extend(uncert_samp_idx)
            
            if bool_ceal == True:
                # get high confidence samples `dh`
                hcs_idx, hcs_labels, hcs_prob = get_high_confidence_samples(pred_prob=pred_prob,
                                                                delta=delta_0)
                # get the original indices
                hcs_idx = [unlabeled_loader.sampler.indices[idx] for idx in hcs_idx]
            
            '''
            if use_cifar == 1:
                # Get classes for the uncertainty samples
                for idx in uncert_samp_idx:
                    label = dataset.targets[idx]
                    classCount[int(label)] += 1
            else:
                # Get classes for the uncertainty samples
                for idx in uncert_samp_idx:
                    label = dataset.dataset.iloc[idx, 0].split(' ')[1]
                    classCount[int(label)] += 1

            fh.write('**** Class count: {} ****\n'.format(classCount))
            '''
            uncert_prob_list.append(uncert_prob[0])
            
            if bool_ceal == True:
                # remove the samples that already selected as uncertain samples.
                hcs_idx = [x for x in hcs_idx if
                        x not in list(set(uncert_samp_idx) & set(hcs_idx))]

                if use_cifar == 1:
                    # add high confidence samples to the labeled set 'dl'
                    # (1) update the indices
                    labeled_loader.sampler.indices.extend(hcs_idx)
                    # (2) update the original labels with the pseudo labels.
                    some_count = 0
                    for idx in range(len(hcs_idx)):
                        if hcs_labels[idx] == labeled_loader.dataset.labels[hcs_idx[idx]]:
                            some_count += 1
                        labeled_loader.dataset.labels[hcs_idx[idx]] = hcs_labels[idx]
                    if len(hcs_idx) > 0:
                        fh.write('hcs_labels: {}'.format(some_count/len(hcs_idx)))

                else:
                    # add high confidence samples to the labeled set 'dl'
                    # (1) update the indices
                    labeled_loader.sampler.indices.extend(hcs_idx)
                    # (2) update the original labels with the pseudo labels.
                    some_count = 0
                    for idx in range(len(hcs_idx)):
                        if hcs_labels[idx] == labeled_loader.dataset.labels[hcs_idx[idx]]:
                            some_count += 1
                        labeled_loader.dataset.labels[hcs_idx[idx]] = hcs_labels[idx]
                    if len(hcs_idx) > 0:
                        fh.write('hcs_labels: {}'.format(some_count/len(hcs_idx)))


                # remove the uncertain samples from the unlabeled pool
            for val in uncert_samp_idx:
                unlabeled_loader.sampler.indices.remove(val)

            fh.write('Update size of labeled and unlabeled dataset by adding {} uncertain samples and {} high certainty samples\n'
                    'updated len(labeled): {}\t updated len(unlabeled): {}\n'.
                    format(len(uncert_samp_idx), len(hcs_idx),len(labeled_loader.sampler.indices),len(unlabeled_loader.sampler.indices)))
            fh.close()

        fh = open(log_file, 'a+')
        fh.write('\nList acc: {}\n'
                 'List balacc: {}\n'
                 'Uncert list: {}\n'
                 'Fraction: {}\n'.format(acc_list, balacc_list, uncert_prob_list,fraction))
        fh.close()
        
        tot_acc = [a + b for a, b in zip(tot_acc, acc_list)]
        tot_balacc = [a + b for a, b in zip(tot_balacc, balacc_list)]
        tot_precision = [a + b for a, b in zip(tot_precision, precision_list)]
        tot_uncert = [a + b for a, b in zip(tot_uncert, uncert_prob_list)]
        tot_train_time = [a + b for a,b in zip(tot_train_time, train_time)]
        tot_len_labeled_samples = [a + b for a,b in zip(tot_len_labeled_samples, len_labeled_samples)]
        stop_flag = 1

    t11 = time.time()
    return tot_acc, tot_balacc, tot_precision, tot_uncert, fraction, t11-t00, tot_train_time, tot_len_labeled_samples
    

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    data_dir = sys.argv[1]+"/"   
    num_classes = int(sys.argv[2]) #7 # DYNAMIC
    batch_size = int(sys.argv[3])
    k_samples = int(sys.argv[4])
    log_file = data_dir + sys.argv[5] + ".log"
    file_ending = sys.argv[6]  #".jpg"
    use_cifar = int(sys.argv[7])
    method = sys.argv[8] # "ceal"/"al"
    criteria = sys.argv[9]

    model_name = "resnet34"
    header_file = data_dir + "header.tfl.txt"
    filename = data_dir + "image_set.data"
    result_file = data_dir + "FINAL-RESULT-10.log"
    size = 64
    num_channels = 3
    epochs = 10  # Add break when training loss stops decreasing 
    bench_epochs = 20
    num_iter = 40
    criterias = ["en", "cl", "ms"]
    delta_0 = 0.0005
    methods = ["ceal", "al"]
    print(use_cifar)

    if use_cifar == 1:
        dataset = load_cifar(log_file)
        print("use cifar")
    else:
        dataset = load_data_pool(data_dir, header_file, filename, log_file, file_ending, num_classes)
        print("dont use cifar")

    bool_ceal = False
    if method == "ceal":
        bool_ceal = True

    tot_acc, tot_balacc, tot_precision, tot_uncert, fraction, tot_time, train_time, tot_len_labeled_samples = run(device, log_file, epochs, batch_size, dataset, num_iter, start_lr, weight_decay, num_classes, criteria, k_samples, model_name, size, num_channels, delta_0, bool_ceal, use_cifar)
    #benchmark(device, log_file, bench_epochs, batch_size, dataset, start_lr, weight_decay, num_classes, model_name, size, num_channels)
    fh = open(result_file, 'a+')
    fh.write('\n**** RESULTS ****\n')
    fh.write('Method: {}\n'.format(method))
    fh.write('Dataset: {} \n'.format(data_dir))
    fh.write('batch size: {}, k_samples: {}, model name: {}, criteria: {}\n'.format(batch_size, k_samples, model_name, criteria))
    #fh.write('criteria: {}\n avg acc: {}\n avg bacc: {}\n avg precision: {}\n avg uncert: {}\n'.format(criteria,  [x/5 for x in tot_acc],  [x/5 for x in tot_balacc], [x/5 for x in tot_precision], [x/5 for x in tot_uncert]))
    fh.write('criteria: {}\n avg acc: {}\n avg bacc: {}\n avg precision: {}\n avg uncert: {}\n'.format(criteria,  [x for x in tot_acc],  [x for x in tot_balacc], [x for x in tot_precision], [x for x in tot_uncert]))
    fh.write('Total time: {}\n Avg train time: {}\n'.format(tot_time, [x for x in train_time]))
    fh.write('Avg len labeled samples: {}\n'.format([x for x in tot_len_labeled_samples]))
    fh.close()


