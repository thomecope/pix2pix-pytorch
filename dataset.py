import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms


class CityscapesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        #self.transforms = self.get_transforms()


    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :256, :]  # check
        target_image = image[:, 256:, :]  # check

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


    def get_transforms(self):
        transform_list = [
            transforms.Resize(286),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(transform_list)
        return transform


if __name__ == "__main__":
    dataset = CityscapesDataset('/Users/amurguia/Documents/EECS545/eecs545project/brain_images_concatenated/train')
    loader = DataLoader(dataset, batch_size=5)
    i = 0
    for x, y in loader:
        print(i)
        i = i + 1
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()