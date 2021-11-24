import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import inception_v3
from evaluate_dataset import EvaluationDataset
from torch.utils.data import DataLoader
from inception_extractor import PartialInceptionNetwork
from scipy.stats import entropy


def convert_numpy(images, batch_size=1):
    l = []
    if images.shape[0] == 1:
        return images[0].permute(1, 2, 0).numpy()
    else:
        for i in range(batch_size):
            img = images[i].permute(1, 2, 0)
            l.append(img)
        return np.array(l)


def show_imgs(img_fake, img_real):
    plt.imshow(img_fake.permute(1, 2, 0))
    plt.title("Fake")
    plt.show()
    plt.imshow(img_real.permute(1, 2, 0))
    plt.title("Real")
    plt.show()

# Pretty sure this is evaluated in the model, but it is repeated here as another metric (not a good one though)
def avg_log_likelihood(img_fake, img_real):  # Method used in original GAN paper (kernel estimation)
    pass


# Set up for having batches of images processed and then averaged to get one unified score
# took inspiration from: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
def inception_score(img_fake_dataset, model, batch_size=32, n_split=10, esp=1E-16):  # Method using inception net for classification
    N = len(img_fake_dataset)  # The number of total images that we have
    fake_loader = DataLoader(img_fake_dataset, batch_size)  # Get a dataloader for iterable batches

    assert N >= batch_size

    model.eval()

    def predict(x):
        return F.softmax(model(x)).detach().numpy()  # Need softmax at the end to get probability dist

    preds = np.zeros((N, 1000))  # There are 1000 output predictions
    for i, imgs in enumerate(fake_loader):
        bsize_i = imgs.size()[0]
        preds[i*batch_size:i*batch_size + bsize_i] = predict(imgs)

    splits = []
    for k in range(n_split):
        part = preds[k * (N // n_split): (k+1) * (N // n_split), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        splits.append(np.exp(np.mean(scores)))

    return np.mean(splits), np.std(splits)


def FID(img_fake, img_real, model):  # activation of last pooling layer of inception net
    # img_fake, img_real are numpy arrays of size (299, 299, 3)
    model.eval()
    output_fake = model(img_fake)
    output_real = model(img_real)
    output_fake = output_fake.detach().numpy().T
    output_real = output_real.detach().numpy().T

    mu1, sig1 = np.mean(output_fake, axis=0), np.cov(output_fake, rowvar=False)
    mu2, sig2 = np.mean(output_real, axis=0), np.cov(output_real, rowvar=False)

    sum_square = np.power(np.linalg.norm(mu1-mu2), 2)

    if output_real.shape[1] == 1:
        mat_sqrt = np.sqrt(sig1*sig2)
    else:
        mat_sqrt = sqrtm(np.matmul(sig1, sig2))

    if np.iscomplexobj(mat_sqrt):
        mat_sqrt = mat_sqrt.real

    return sum_square + np.trace(sig1 + sig2 - np.multiply(mat_sqrt, 2)) if output_real.shape[1] != 1 \
        else sum_square + sig1 + sig2 - np.multiply(mat_sqrt, 2)






if __name__ == "__main__":
    # Gather models needed for IC and FID
    inception_net = inception_v3(pretrained=True)
    inception_net.eval()  # Set up the model for evaluation
    inception_act = PartialInceptionNetwork()
    inception_act.eval()

    # Gather datasets based on custom dataset class
    real_dataset = EvaluationDataset(root_dir="data/images/gan/real")
    fake_dataset = EvaluationDataset(root_dir="data/images/gan/fake")

    # Get the number of images that belong to each
    num_images = len(real_dataset)

    # Get the dataloaders --> We can choose how many to average over with the batch size
    real_loader = DataLoader(real_dataset, batch_size=5, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=5, shuffle=False)

    # Gather a big batch of the data (can use for testing, not necessary with loop)
    img_real_batch = next(iter(real_loader))
    img_fake_batch = next(iter(fake_loader))
    # print(np.mean(inception_act(img_fake_batch).detach().numpy()))
    ALL_dict = {}
    IC_dict = {}
    FID_dict = {}
    print(inception_score(fake_dataset, inception_net))
    # print(FID(img_fake_batch, img_real_batch, inception_act))
# For actual training evaluation, we would gather all of the images in one batch and loop through epochs
    uncomment = """
    for real, fake in zip(real_loader, fake_loader):
        # all_score = avg_log_likelihood(fake, real)
        # ic_score = inception_score(fake, real, inception_net)
        fid_score = FID(fake, real, inception_act)

        # ALL.append(avg_log_likelihood(fake, real))
        # IC.append(inception_score(fake, real, inception_net))
        FID_dict[round(fid_score, 4)] = (real, fake)

    print(FID_dict)"""

