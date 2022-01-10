import pytest
import sys
import os
import torch

sys.path.append(os.getcwd()+'/src/models')
from dataset_fetcher import Dataset_fetcher
from model import MyAwesomeModel

model=MyAwesomeModel()

train_img_path='data/processed/corruptmnist/train_images.pt'
train_lab_path='data/processed/corruptmnist/train_labels.pt'
test_img_path='data/processed/corruptmnist/test_images.pt'
test_lab_path='data/processed/corruptmnist/test_labels.pt'
trainset=Dataset_fetcher(train_img_path,train_lab_path)
testset=Dataset_fetcher(test_img_path,test_lab_path)

def test_input():
    assert model.fc1.in_features==torch.mul(*trainset.images.shape[-2:]).detach().numpy().tolist()
