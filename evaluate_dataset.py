from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class EvaluationDataset(Dataset):
    def __init__(self, root_dir, model="Inception"):
        self.root_dir = root_dir
        self.model = model
        self.list_files = sorted(os.listdir(self.root_dir))
        self.transforms = self.get_transforms()
        print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, item):
        img_file = self.list_files[item]
        img_path = os.path.join(self.root_dir, img_file)
        image_pil = np.array(Image.open(img_path))
        # print(image_pil.shape)
        # Hardcoded until we guarantee the output size
        segment = Image.fromarray(image_pil[:, :256, :])
        generated = Image.fromarray(image_pil[:, 256:516, :])
        real = Image.fromarray(image_pil[:, 516:, :])
        # image = self.transforms(image_pil)
        return self.transforms(segment), self.transforms(generated), self.transforms(real)

    def get_transforms(self):
        if self.model == "Inception":
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

        if self.model == "FCN":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225]), ])
            return transform

    def get_list(self):
        return self.list_files



if __name__ == "__main__":
    # Test to make sure it runs with no errors
    eval_test = EvaluationDataset(root_dir="sample_images")
    segment, generated, real = next(iter(eval_test))

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
    print(segment_2.shape)  # Should output (batch_size, 3 channels, 299, 299 if Inception)