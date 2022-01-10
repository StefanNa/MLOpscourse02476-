import torch
from torch.utils.data import Dataset



class Dataset_fetcher(Dataset):
    def __init__(self,PATH_IMG,Path_LAB,transform=None):
        self.transform=transform
        self.images=torch.load(PATH_IMG)
        self.labels=torch.load(Path_LAB)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x,y

    def __len__(self):
        return (len(self.images))