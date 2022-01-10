import pytest
import sys
import os
import torch

sys.path.append(os.getcwd()+'/src/models')
from dataset_fetcher import Dataset_fetcher




# class dataloader_tests:
#     def __init__(self):
#         train_img_path='data/processed/corruptmnist/train_images.pt'
#         train_lab_path='data/processed/corruptmnist/train_labels.pt'
#         test_img_path='data/processed/corruptmnist/test_images.pt'
#         test_lab_path='data/processed/corruptmnist/test_labels.pt'
#         self.trainset=Dataset_fetcher(train_img_path,train_lab_path)
#         self.testset=Dataset_fetcher(test_img_path,test_lab_path)
#     def test_length(self):
#         assert [len(self.trainset),len(self.testset)]==[25000, 5000]
    
#     def test_shape(self):
#         assert self.trainset.images.shape[-2:]==self.testset.images.shape[-2:]==torch.Size([28,28])


train_img_path='data/processed/corruptmnist/train_images.pt'
train_lab_path='data/processed/corruptmnist/train_labels.pt'
test_img_path='data/processed/corruptmnist/test_images.pt'
test_lab_path='data/processed/corruptmnist/test_labels.pt'
trainset=Dataset_fetcher(train_img_path,train_lab_path)
testset=Dataset_fetcher(test_img_path,test_lab_path)

def test_length():
    assert [len(trainset),len(testset)]==[25000, 5000]

def test_shape():
    assert trainset.images.shape[-2:]==testset.images.shape[-2:]==torch.Size([28,28])

def test_labels():
    assert torch.unique(trainset.labels).shape==torch.unique(testset.labels).shape==torch.Size([10])

# assert [len(trainset),len(testset)]==[25000, 5000]


# print((trainset.images.shape[-2:]==testset.images.shape[-2:]==torch.Size([28,28])))