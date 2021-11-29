# EECS 545 Fall 2021
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class mri_dataset:
    """
    Dog Dataset.
    """
    # def __init__(self, batch_size=4, dataset_path='brain_images_concatenated', if_resize=True):
    #     self.batch_size = batch_size
    #     self.dataset_path = dataset_path
    #     self.if_resize = if_resize
    #     #self.train_dataset = self.get_train_numpy()
    #     self.x_mean, self.x_std = self.compute_train_statistics()
    #     self.transform = self.get_transforms()
    #     self.train_loader, self.val_loader = self.get_dataloaders()


    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.transforms = self.get_transforms()
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, item):
        img_file = self.list_files[item]
        img_path = os.path.join(self.root_dir, img_file)
        image_pil = Image.open(img_path)
        image = self.transforms(image_pil)
        return image

    def get_train_numpy(self):
        train_dataset = torchvision.datasets.ImageFolder(os.path.join('/Users/amurguia/Documents/EECS545/eecs545project', self.dataset_path, 'train'))
        train_x = np.zeros((len(train_dataset), 224, 224, 3))
        # train_x = np.zeros((len(train_dataset), 64, 64, 3))
        for i, (img, _) in enumerate(train_dataset):
            train_x[i] = img
        return train_x / 255.0

    def compute_train_statistics(self):
        # TODO (part a): compute per-channel mean and std with respect to self.train_dataset
        x_mean = np.mean(self.train_dataset,(0,1,2))  # per-channel mean
        x_std = np.std(self.train_dataset,(0,1,2))  # per-channel std
        return x_mean, x_std

    def get_transforms(self):
        if self.if_resize:
            # TODO (part a): fill in the data transforms
            transform_list = [ 
                # resize the image to 32x32x3
                #transforms.Resize([32,32]), 
                # convert image to PyTorch tensor
                transforms.ToTensor(),
                # normalize the image (use self.x_mean and self.x_std)
                #transforms.Normalize(self.x_mean, self.x_std),
                #transforms.CenterCrop([224,224])
            ]

        transform = transforms.Compose(transform_list)
        return transform

    def get_dataloaders(self):
        # train set
        train_set = torchvision.datasets.ImageFolder('/Users/amurguia/Documents/EECS545/eecs545project/brain_images_concatenated/train', transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # validation set
        val_set = torchvision.datasets.ImageFolder('/Users/amurguia/Documents/EECS545/eecs545project/brain_images_concatenated/val', transform=self.transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    # def plot_image(self, image, label):
    #     image = np.transpose(image.numpy(), (1, 2, 0))
    #     image = image * self.x_std.reshape(1, 1, 3) + self.x_mean.reshape(1, 1, 3)  # un-normalize
    #     plt.title(label)
    #     plt.imshow((image*255).astype('uint8'))
    #     plt.show()

    # def get_semantic_label(self, label):
    #     mapping = {'chihuahua': 0, 'dalmatian': 1, 'golden_retriever': 2, 'samoyed': 3, 'siberian_husky': 4}
    #     reverse_mapping = {v: k for k, v in mapping.items()}
    #     return reverse_mapping[label]


if __name__ == '__main__':
    dataset = mri_dataset(root_dir='/Users/amurguia/Documents/EECS545/eecs545project/brain_images_concatenated/train')
    #dataset = mri_dataset()
    print(dataset.train_dataset.shape)
    images, labels = iter(dataset.train_loader).next()
    # dataset.plot_image(
    #     torchvision.utils.make_grid(images),
    #     ', '.join([dataset.get_semantic_label(label.item()) for label in labels])
    #)
