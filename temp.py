from dataloader_ import ImageDataset 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch 

dataset = ImageDataset(r"train_input_2k/train_input_2k" , r"train_gt_2k/train_gt_2k")
dataloader = DataLoader(dataset, batch_size=16 , shuffle=True)


for (x , y) in dataloader:
    print(x.shape , y.shape)