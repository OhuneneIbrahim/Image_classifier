import pandas as pd
import torch 
from torch import nn
from torch import optim 
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import json
import matplotlib
matplotlib.use('Agg')
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
parser.add_argument('--test_input', type= str, action = 'store',
                    default='./flowers/test/12/image_04016.jpg', 
                    help ='enter the directory for the test image')
parser.add_argument('checkpoint', type = str, action = 'store', default = './model_checkpoint.pth',
                   help ='save model to checkpoint')
parser.add_argument('--topk', type = int, action = 'store', default = 3,
                   help = 'specifiy the topk classes')
parser.add_argument('--gpu',action ='store_true', default = True, 
                   help ='if true, gpu will be used else cpu is used')
parser.add_argument('--category',action ='store', default = 'cat_to_name.json', 
                   help = 'map catogories to real name')

flag = parser.parse_args()
directory = flag.data_dir
checkpoint_path = flag.checkpoint
inputs= flag.test_input
Topk = flag.topk
Cat = flag.category
gpu_flag = flag.gpu


def main():
    model =sv.load_checkpoint(checkpoint_path)
    if gpu_flag and torch.cuda.is_available(): 
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"      ******************** {device} environment has been activated **********************")
    print("                                         ...")
    print("                                         ...")
    print("                                         ...")
    
    print(f"           ******************** model and dataset moved to {device} **********************")
    print("                                 ...")
    print("                                 ...")
    print("                                 ...")
    #print("\n")
    with open(Cat, 'r') as f:
        cat_to_name = json.load(f)
    model.to(device)
    image_path = inputs 
    prob, classes, image = prc.predict(image_path, model, Topk, device)

    class_name = [cat_to_name[str(element)] for element in classes]
    
    print(class_name)
    print("\nFlower name (probability): ")
    print("*********")
    for i in range(len(prob)):
        print(f"{class_name[i]} ------------- ({prob_list[i]})")
    print("")
   
    
    print(f"********************************** View your flower image and name *************************************") 
    
    figure, (axis_1, axis_2) = plt.subplots(2, 1, figsize=(5, 5))

    print(f"prob:{prob}--classes:{classes}")
    #image = process_image(path)
    imshow(image, ax=axis_1)
    axis_1.set_title(cat_to_name[str(12)])
    class_name = [cat_to_name[str(i)] for i in classes]


    ticks = range(len(classes))
    div_ = (max(probs)-min(probs))/5.
    val_ = np.arange(min(probs), max(probs)+.1, div_)

    axis_2.set_xticks(val_)
    axis_2.set_yticks(ticks)
    axis_2.set_yticklabels(class_name)
    axis_2.set_xlabel('Prediction probability')
    axis_2.invert_yaxis()
    axis_2.barh(ticks, probs)


    plt.savefig('plot.png')

if __name__== "__main__":
    main()