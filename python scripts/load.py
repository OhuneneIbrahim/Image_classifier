from collections import OrderedDict
import pandas as pd
import torch 
from torch import nn
from torch import optim 
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import process as prc


def load_data(root_dir = 'flowers'):
   
    data_dir = root_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    
    train_transforms = prc.train_transform()
    valid_transforms = prc.test_transform()
    test_transforms = prc.test_transform()
    
    #  Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms) 
    validate_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform= test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=64)
    test_loader=torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_loader, validate_loader, test_loader, train_data