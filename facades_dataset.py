from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class FacadeDataset(Dataset):
    def __init__(self, root_dir, isSegment):
        self.root_dir = root_dir
        self.isSegment = isSegment
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
        return torch.round(image.squeeze(0)*255).long() if self.isSegment else image

    def get_transforms(self):
        if self.isSegment:
            transform_list = [
                transforms.ToTensor(),
                transforms.Resize((256, 256))

            ]
            transform = transforms.Compose(transform_list)
            return transform

        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((256, 256))
            ]
            transform = transforms.Compose(transform_list)
            return transform


test = """if __name__ == "__main__":
    # Test to make sure it runs with no errors
    eval_test = FacadeDataset(root_dir="sample_images", isSegment=False)
    print(next(iter(eval_test)))

    test = 
    print(type(segment))

    plt.imshow(segment.permute(1, 2, 0))
    plt.show()

    plt.imshow(generated.permute(1, 2, 0))
    plt.show()

    plt.imshow(real.permute(1, 2, 0))
    plt.show()

    # Test it on the Dataloader

    eval_test_loader = DataLoader(eval_test, batch_size=1, shuffle=False)
    segment_2, generated_2, real_2 = next(iter(eval_test_loader))
    print(segment_2.shape)  # Should output (batch_size, 3 channels, 299, 299 if Inception)"""