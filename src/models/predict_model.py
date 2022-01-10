import argparse
import sys

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset

from dataset_fetcher import Dataset_fetcher as DSF
from model import MyAwesomeModel
import numpy as np
import os
import matplotlib.pyplot as plt


def predict():
    parser = argparse.ArgumentParser(description='prediction arguments')
    parser.add_argument("--c",'-checkpoint',type=str, default='../../models/processed/corruptmnist/checkpoint.pth',help="checkpoint directory")
    parser.add_argument("--images",type=str, default='../../data/processed/corruptmnist/test_images.pt',help="predict images path .pt")
    args = parser.parse_args(sys.argv[2:])
    checkpoint_path=args.c
    images_path=args.images

    images=torch.load(images_path)
    random_image=np.random.randint(images.shape[0])
    image_size=images.shape[-2:]
    sample_image=images[random_image].view(-1,*image_size)

    model = MyAwesomeModel()
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    with torch.no_grad():
        # set model to evaluation mode
        model.eval()
    log_ps = model(sample_image)
    ps = torch.exp(log_ps)
    prob,pred=torch.max(ps,1)
    prob=prob.detach().numpy()[0]
    pred=pred.detach().numpy()[0]
    print('class: ',pred,', Probability: ',prob)
    return pred,prob

if __name__ == '__main__':
    
    predict()


