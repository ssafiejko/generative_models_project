import os
import random
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3


class CatsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob(os.path.join(root_dir, '*.png'))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Return a dummy label (0) since we don't have classes
        return image, 0

def make_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds = CatsDataset(cfg.root_path, transform=transform)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    return loader