import pandas as pd
import torch 
from torch import nn
from torch import optim 
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import logging
import pathlib





def train_transform():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
    return train_transforms

def test_transform():
    test_transforms = transforms.Compose([transforms.Resize(225),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])
                                    ])
    return test_transforms


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #  Process a PIL image for use in a PyTorch model
    image_transform = test_transform() 
    with Image.open(image) as im:
        image = image_transform(im)
    return image



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    Processed_img = process_image(image_path)
    numpy_img = Processed_img.numpy()
    batched_numpy_img = np.array([numpy_img])
    pytorch_img = torch.from_numpy(batched_numpy_img).float()

    with torch.no_grad():
        img=pytorch_img.cuda()
        logps = model.forward(img)
        
    prediction = torch.exp(logps).data
    prob, classes=prediction.topk(topk)
    top_indices = np.array(prob[0].cpu().numpy())
    classes = np.array(classes.cpu().numpy())
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    top_classes = np.array([idx_to_class[x] for x in classes[0]])
    return top_indices, top_classes, Processed_img