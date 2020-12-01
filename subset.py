
from torch.utils.data import Dataset
import torch 

class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = dataset
        self.indices = indices
       # labels_hold = torch.ones(len(dataset)).type(torch.long) *300 #( some number not present in the #labels just to make sure
       # labels_hold[self.indices] = labels 
        self.labels = labels
    def __getitem__(self, idx):
         return self.dataset[self.indices[idx]]


    def __len__(self):
        return len(self.indices)