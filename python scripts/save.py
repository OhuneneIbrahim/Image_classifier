from collections import OrderedDict
import pandas as pd
import torch 
from torch import nn
from torch import optim 
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import model as mdl


def checkpoint(train_data, checkpoint_path = 'model_checkpoint.pth',
              model_arch = "vgg16", hidden_layer_size = 500, drop =0.2, lr = 0.001, epoch = 5):
    model, _,_, _, _= mdl.model_setup(model_arch, hidden_layer_size, drop, lr)
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'model_Arch': model_arch,
              'classifier_input_size': 25088,
              'classifier_hidden_size':hidden_layer_size,
              'learning_rate':lr,
              'criterion':nn.NLLLoss(),
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              #'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'Dropout_propability':drop

   }
    torch.save(checkpoint, checkpoint_path)
    
    
    
def load_checkpoint(checkpoint_path = 'model_checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    hidden_layer_size = checkpoint['classifier_hidden_size']
    model_arch = checkpoint['model_Arch']
    lr = checkpoint['learning_rate']
    criterion = checkpoint['criterion'] 
    epoch = checkpoint['epoch']
    drop = checkpoint ['Dropout_propability']
    model,_,_,_,_ = mdl.model_setup(model_arch, hidden_layer_size, drop, lr)
   
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    #optimizer_state_dict = checkpoint['optimizer_state_dict']
    model.eval()
    return model