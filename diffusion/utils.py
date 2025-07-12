import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
import random
import math
import pdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000, device: torch.device = None):
        super().__init__()
        beta = torch.linspace(1e-4, 0.02, num_time_steps, device=device)
        alpha = 1 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha_cumprod)

    def forward(self, t):
        return self.beta[t], self.alpha[t]

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        return self

class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def main():
    scheduler = DDPM_Scheduler(num_time_steps=1000)
    print(scheduler(999))

if __name__ == '__main__':
    main()
