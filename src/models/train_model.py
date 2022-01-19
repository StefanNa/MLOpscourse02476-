import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from dataset_fetcher import Dataset_fetcher as DSF
from model import MyAwesomeModel
from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))

def saveplot(loss,PATH='../../reports/figures/'):
    plt.plot([i for i in range(len(loss))],loss)
    plt.xlabel('epochs')
    plt.ylabel('running loss')
    plt.savefig(PATH+'loss_curve.png')

def _power_of_2(num):
    """Check if number is power of 2"""

    cond = np.log2(int(num))

    if np.ceil(cond) != np.floor(cond):
        raise argparse.ArgumentTypeError("Argument must be a power of 2")

    return int(num)

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        #get data
        
        try:
            parser = argparse.ArgumentParser(
                description="Script for either training or evaluating",
                usage="python main.py <command>"
            )

            parser.add_argument(
                "command", choices=["train", "evaluate"], help="Subcommand to run"
            )
            
            args = parser.parse_args(sys.argv[1:2])

            if not hasattr(self, args.command):
                print('Unrecognized command')
                
                parser.print_help()
                exit(1)
            
            
            # use dispatch pattern to invoke method with same name
            getattr(self, args.command)()
        except:
            self.train()
        
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        # parser.add_argument("--c",'-checkpoint',type=str, default='models/processed/corruptmnist/',help="checkpoint directory")
        # parser.add_argument("--cname",'-checkpoint_filename',type=str, default='checkpoint.pth',help="checkpoint filename -- checkpoint.pth")
        parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
        parser.add_argument("--e", type=int, default=10, help="epoch")
        # parser.add_argument("--b", type=_power_of_2, default=64, help="batch size")
        # parser.add_argument("--cont", type=str, default=None, help="Path to model that should be trained again")
        # parser.add_argument("--PATH_IMG",type=str, default='data/processed/corruptmnist/train_images.pt',help="path to images")
        # parser.add_argument("--PATH_LAB", type=str, default='data/processed/corruptmnist/train_labels.pt', help="Path to labels")

        args = parser.parse_args(sys.argv[2:])
        print(args)


        checkpoint='models/dict/corruptmnist/'#args.c
        picklepath='models/pickle/corruptmnist/'
        checkpoint_name='checkpoint.pth'#args.cname
        lr=args.lr
        epochs=args.e
        batchsize=64#args.b
        PATH_IMG='data/processed/corruptmnist/train_images.pt'#args.PATH_IMG
        PATH_LAB='data/processed/corruptmnist/train_labels.pt'#args.PATH_LAB

        if checkpoint[-1] != '/':
            checkpoint.append('/')

        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        if not os.path.exists(picklepath):
            os.makedirs(picklepath)
        #get dataset
        dataset=DSF(PATH_IMG,PATH_LAB)

        # split train set in test and validation set
        validation_split = .2
        seed = 42
        train_indices, validation_indices, _, _ = train_test_split(
            range(len(dataset)),
            dataset.labels,
            stratify=dataset.labels,
            test_size=validation_split,
            random_state=seed
        )
        trainset = Subset(dataset, train_indices)
        validationset = Subset(dataset, validation_indices)

        trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
        validloader = DataLoader(validationset, batch_size=batchsize, shuffle=True)
        # TODO: Implement training loop here
   
        model = MyAwesomeModel()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses=[]
        best_accuracy=0
        for e in range(epochs):
            running_loss=0
            model.train()
            for images, labels in trainloader:
                optimizer.zero_grad()
        
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            train_losses.append(running_loss)
            # print(running_loss)
            if e%3==0:
                model.eval()
                
                with torch.no_grad():
                # set model to evaluation mode
                    model.eval()
                accuracies=[]
                for images, labels in validloader:
                    log_ps = model(images)
                    ps = torch.exp(log_ps)
                    pred=torch.max(ps,1)[1]
                    correct= pred==labels.view(*pred.shape)
                    accuracy=correct.type(torch.FloatTensor).mean()
                    accuracies.append(accuracy.item()*100)
                print(f'Accuracy: {np.mean(accuracies)}%')
                if best_accuracy<np.mean(accuracies):
                    best_accuracy=np.mean(accuracies)
                    torch.save(model,picklepath+'model.pth')
                    torch.save(model.state_dict(), checkpoint+checkpoint_name)

        if not os.path.isdir('reports/figures/'):
            os.makedirs('reports/figures/')
        saveplot(train_losses,PATH='reports/figures/')

        bucket_name="mlops-project-6"
        source_file_checkpoint=checkpoint+"checkpoint.pth"
        source_file_pickle=picklepath+"model.pth"
        destination_blob_name="mnist/models/"

        upload_blob(bucket_name, source_file_pickle, destination_blob_name+'pickle/model.pth')
        upload_blob(bucket_name, source_file_checkpoint, destination_blob_name+'dict/checkpoint.pth')

        
    def evaluate(self):
        # print("Evaluating until hitting the ceiling")
        # parser = argparse.ArgumentParser(description='Training arguments')
        # parser.add_argument('model_directory', default="models/processed/corruptmnist/")
        # parser.add_argument('model_filename', default="checkpoint.pth")
        # parser.add_argument("--PATH_IMG",type=str, default='data/processed/corruptmnist/train_images.pt',help="path to images")
        # parser.add_argument("--PATH_LAB", type=str, default='data/processed/corruptmnist/train_labels.pt', help="Path to labels")

        # # add any additional argument that you want
        # args = parser.parse_args(sys.argv[2:])
        # print(args)
        
        model_directory="models/processed/corruptmnist/"#args.model_directory
        model_filename="checkpoint.pth"#args.model_filename
        PATH_IMG='data/processed/corruptmnist/train_images.pt'#args.PATH_IMG
        PATH_LAB='data/processed/corruptmnist/train_labels.pt'#args.PATH_LAB


        testset=DSF(PATH_IMG,PATH_LAB)
        testloader = DataLoader(testset, batch_size=64, shuffle=True)

        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        state_dict = torch.load(model_directory+model_filename)
        model.load_state_dict(state_dict)

        # torch.load(args.load_model_from)
        # _, test_set = mnist()

        with torch.no_grad():
            # set model to evaluation mode
            model.eval()
        accuracies=[]
        
        for images, labels in testloader:
            log_ps = model(images)
            ps = torch.exp(log_ps)
            pred=torch.max(ps,1)[1]
            correct= pred==labels.view(*pred.shape)
            accuracy=correct.type(torch.FloatTensor).mean()
            accuracies.append(accuracy.item()*100)
        print(f'Accuracy: {np.mean(accuracies)}%')


if __name__ == '__main__':
    
    TrainOREvaluate()
    #python3 train_model.py train --lr=0.003

    #python3 train_model.py evaluate --PATH_IMG=../../data/processed/corruptmnist/test_images.pt --PATH_LAB=../../data/processed/corruptmnist/test_labels.pt ../../models/processed/corruptmnist/ checkpoint.pth 
