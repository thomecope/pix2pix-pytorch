from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class EvaluationDataset(Dataset):
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

    def get_transforms(self):
        transform_list = [
            # These were specified for the inception net model
            # https://pytorch.org/hub/pytorch_vision_inception_v3/
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            # For these images, the normalization causes it to appear all white, consider changing?
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        transform = transforms.Compose(transform_list)
        return transform


if __name__ == "__main__":
    # Test to make sure it runs with no errors
    data_fake = EvaluationDataset(root_dir="data/images/gan/fake")
    data_fake_generator = DataLoader(data_fake, batch_size=1, shuffle=False)
    img_fake = next(iter(data_fake_generator))[0]
    plt.imshow(img_fake.permute(1, 2, 0))
    plt.show()

    data_real = EvaluationDataset(root_dir="data/images/gan/real")
    data_real_generator = DataLoader(data_real, batch_size=2, shuffle=False)
    img_real = next(iter(data_real_generator))[0]
    plt.imshow(img_real.permute(1, 2, 0))
    plt.show()

    # For some reason the code below doesnt work in Pycharm?
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(img_fake.permute(1, 2, 0))
    axs[0].set_title('Fake')
    axs[1].imshow(img_real.permute(1, 2, 0))
    axs[1].set_title('Real')

    #fig.show()