import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
import scipy

# from https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/master/fid.py


class PartialInceptionNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)  # Grabs the inception net
        # The following line makes sure that we get the output from the final convolution (b4 avg pooling)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        # x = x * 2 -1 # Normalize to [-1, 1]  Need to do more research into this

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations
