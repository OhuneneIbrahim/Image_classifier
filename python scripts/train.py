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
import process as prc
import model as mdl
import load as ld
import save as sv

parser = argparse.ArgumentParser(description='parser argument specification for the train.py script')

parser.add_argument('data_dir', type= str, action = 'store', default='./flowers/', 
                    help ='enter the root directory')
parser.add_argument('--save_dir', type = str, action = 'store', default = './model_checkpoint.pth',
                   help ='save model to checkpoint')
parser.add_argument('--arch', type = str, action = 'store', default = 'vgg16',
                   help = 'select the model architecture')
parser.add_argument('--learning_rate', type= float, action = 'store', default = 0.001,
                   help = 'specify a learning rate')
parser.add_argument('--hidden_units', type= int, action = 'store', default = 500,
                   help = 'Specify the number of hidden layer')
parser.add_argument('--epoch', type = int, action = 'store', default = 5,
                   help = 'specify the number of epoch')
parser.add_argument('--gpu',action ='store_true', default = True, 
                   help ='if true, gpu will be used else cpu is used')
parser.add_argument('--dropout',action ='store', default = 0.2)

flag = parser.parse_args()
space = flag.data_dir
path = flag.save_dir
model_arch = flag.arch
LR = flag.learning_rate
hidden_layer_size = flag.hidden_units
epochs = flag.epoch
gpu_flag = flag.gpu
drop= flag.dropout


    
    
def main():
    if gpu_flag and torch.cuda.is_available(): 
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"      ******************** {device} environment has been activated **********************")
    print("                                         ...")
    print("                                         ...")
    print("                                         ...")
    #print("\n")
    train_loader, valid_loader, test_loader, train_data = ld.load_data(space)
    model, criterion, optimizer, lr, Arch = mdl.model_setup(model_arch, hidden_layer_size, drop, LR)
    print(f"           ******************** model and dataset moved to {device} **********************")
    print("                                 ...")
    print("                                 ...")
    print("                                 ...")
    #print("\n")
    model.to(device)
    #training 
    print("      ************************* Model training *******************************************")
    print("                                 ...")
    print("                                 ...")
    print("                                 ...")
    #print("\n")
    
    #epochs = 5
    steps = 0
    print_on = 5
    for epoch in range(epochs):
        running_loss = 0
        for image, label in train_loader:
            steps +=1
            #print(f"******************** moving dataset to {device} **********************")
            image, label = image.to(device), label.to(device)
        
        
            train_image = model.forward(image)
            loss= criterion(train_image, label)
            running_loss += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
            if steps % print_on ==0:
                validateloss=0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for image, label in valid_loader:
                        image, label = image.to(device), label.to(device)
                        val_image = model.forward(image)
                        val_loss=criterion(val_image, label)
                    
                        validateloss +=val_loss.item()
                    
                    #Accuracy
                    
                        ps = torch.exp(val_image)
                        top_p, top_class = ps.topk(1, dim=1)
                        equal = top_class == label.view(*top_class.shape)
                        accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
                    
            
           # train_loss.append(running_loss/print_on)
            #validate_loss.append(validateloss/len(validate_loader.dataset))
                print(f"Epoch {epoch +1}/{epochs}.. "
                    f"Train loss: {running_loss/print_on:.3f}.. "
                    f"Test loss: {validateloss/len(valid_loader):.3f}.. "
                    f"Test accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
    
    print(f"******************** Training completed **********************") 
    print("                         ...")
    print("                         ...")
    print("                         ...")
    
    
    print(f"******************** Model Testing **********************") 
    print("                         ...")
    print("                         ...")
    print("                         ...")
    
    testloss=0
    test_accuracy = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            test_image = model.forward(image)
            test_loss=criterion(test_image, label)
                    
            testloss +=test_loss.item()
                    
                    #Accuracy
                    
            ps = torch.exp(test_image)
            top_p, top_class = ps.topk(1, dim=1)
            equal = top_class == label.view(*top_class.shape)
            test_accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
    print(f"Test accuracy: {test_accuracy/len(test_loader):.3f}")
    
    
    print(f"******************** Saving model **********************")
    print("                       ...")
    print("                       ...")
    print("                       ...")
    sv.checkpoint(train_data, path, model_arch, hidden_layer_size, drop, lr, epoch)   
    print(f"******************** model saved to checkpoint **********************")
if __name__ == "__main__":
    main()