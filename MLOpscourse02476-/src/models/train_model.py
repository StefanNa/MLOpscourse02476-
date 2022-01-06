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
from sklearn.model_selection import train_test_split

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
        ##get data
        
        
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
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument("--c",'-checkpoint',type=str, default='../../model/processed/corruptmnist/',help="checkpoint directory")
        parser.add_argument("--cname",'-checkpoint_filename',type=str, default='checkpoint.pth',help="checkpoint filename -- checkpoint.pth")
        parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
        parser.add_argument("--e", type=int, default=10, help="epoch")
        parser.add_argument("--b", type=_power_of_2, default=64, help="batch size")
        parser.add_argument("--cont", type=str, default=None, help="Path to model that should be trained again")
        parser.add_argument("--PATH_IMG",type=str, default='../../data/processed/corruptmnist/train_images.pt',help="path to images")
        parser.add_argument("--PATH_LAB", type=str, default='../../data/processed/corruptmnist/train_labels.pt', help="Path to labels")

        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        epochs=args.e
        lr=args.lr
        batchsize=args.b
        checkpoint=args.c
        continue_training=args.cont
        checkpoint_name=args.cname
        PATH_IMG=args.PATH_IMG
        PATH_LAB=args.PATH_LAB

        if checkpoint[-1] != '/':
            checkpoint.append('/')

        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        
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

        if continue_training is None:
            True
        else:
            state_dict = torch.load(continue_training)
            model.load_state_dict(state_dict)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses=[]

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
            model.eval()
            # print(running_loss)
            if e%3==0:
                torch.save(model.state_dict(), checkpoint+checkpoint_name)
                # torch.save(model.state_dict(), 'checkpoint'+str(e)+'.pth')
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
        saveplot(train_losses,PATH='../../reports/figures')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('model_directory', default="../../model/processed/corruptmnist/")
        parser.add_argument('model_filename', default="checkpoint.pth")
        parser.add_argument("--PATH_IMG",type=str, default='../../data/processed/corruptmnist/train_images.pt',help="path to images")
        parser.add_argument("--PATH_LAB", type=str, default='../../data/processed/corruptmnist/train_labels.pt', help="Path to labels")

        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        model_directory=args.model_directory
        model_filename=args.model_filename
        PATH_IMG=args.PATH_IMG
        PATH_LAB=args.PATH_LAB


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

    #python3 train_model.py evaluate --PATH_IMG=../../data/processed/corruptmnist/test_images.pt --PATH_LAB=../../data/processed/corruptmnist/test_labels.pt ../../model/processed/corruptmnist/ checkpoint.pth 