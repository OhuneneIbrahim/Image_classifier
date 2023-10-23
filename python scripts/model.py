from collections import OrderedDict
import pandas as pd
import torch 
from torch import nn
from torch import optim 
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def model_setup(model_arch="vgg16", hidden_layer_size = 500, drop=0.2, lr= 0.001):
    if model_arch == "vgg13":
        model = models.vgg13(pretrained = True)
    elif model_arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_arch == "vgg19":
        model = models.vgg16(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad= False
        
    classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(25088, hidden_layer_size)),
                               ("relu", nn.ReLU()),
                                ("Dropout", nn.Dropout(p=drop)),
                                ("fc2", nn.Linear(hidden_layer_size, 256)),
                                ("relu", nn.ReLU()),
                                ("Dropout", nn.Dropout(p=drop)),
                               ("fc3", nn.Linear(256, 102)),
                              ("output", nn.LogSoftmax(dim=1))]))
    
    
    model.classifier = classifier
    #model = model_paramaters()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    print("         ******************** Model setup parameters ********************")
    
    print(f"Model Architecture:    {model_arch} \n")
    print(f"Criterion:             {criterion}\n")
    print(f"Optimizer:             Adam\n")
    print(f"Learning rate:         {lr}\n")
   # 
    return model, criterion, optimizer, lr, model_arch



