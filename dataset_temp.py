# tommy cope

import os
import matplotlib.pyplot as plt   
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random

class Pix2pix_Dataset(Dataset):
    def __init__(self, root_dir, t_flag, grayscale=False):
        '''Initialize directory, transforms'''
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.t_flag = t_flag
        self.grayscale=grayscale
        self.inchannels = 1 if self.grayscale else 3

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_files)

    def __getitem__(self, index):
        '''Generates one sample of data at given index'''
        
        # Select sample and open
        file = self.list_files[index]
        image = np.array(Image.open(os.path.join(self.root_dir, file)))

        # Get inp and tar images
        w = image.shape[1]
        w = w // 2

        inp = Image.fromarray(image[:, w:, :])
        tar = Image.fromarray(image[:, :w, :])

        # Do transforms
        if self.grayscale and self.t_flag:
            gscale = transforms.Grayscale(1)
            inp = gscale(inp)
            tar = gscale(tar)

        if self.t_flag:
            resize = transforms.Resize(286)
            inp = resize(inp)
            tar = resize(tar)

            i, j, h, w = transforms.RandomCrop.get_params(inp, output_size=(256, 256))
            inp = F.crop(inp, i, j, h, w)
            tar = F.crop(tar, i, j, h, w)

        if self.t_flag:
            if random.random() > 0.5:
                inp = F.hflip(inp)
                tar = F.hflip(tar)
        
        tensor = transforms.ToTensor()
        inp = tensor(inp)
        tar = tensor(tar)

        if self.grayscale:
            normalize = transforms.Normalize((0.5,), (0.5,))
            inp = normalize(inp)
            tar = normalize(tar)
        else:
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            inp = normalize(inp)
            tar = normalize(tar)

        return inp, tar

from torch.utils.data import DataLoader

if __name__ == "__main__":
    training_data = Pix2pix_Dataset('./facades/train')
    train_loader = DataLoader(training_data, batch_size=1,shuffle=True)
    x, y = next(iter(train_loader))

    x_img = torch.permute(x[0].squeeze(), (1,2,0))
    y_img = torch.permute(y[0].squeeze(), (1,2,0))

    f, ax = plt.subplots(1,2)
    ax[0].imshow(x_img*0.5+0.5)
    ax[1].imshow(y_img*0.5+0.5)
    plt.show()