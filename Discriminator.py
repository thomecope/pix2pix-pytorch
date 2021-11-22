#Discriminator Model

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

### Used to build discriminator made up of several conv -> BN -> relu 

class Conv_BN_Relu_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        
        ### nn.sequential allows you to concatenate several layers together  

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = stride, bias = False, padding_mode = "reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)

        )


    def forward(self, x):
        return self.block(x)

#Send in generated image x as well as "true" image y along channel dimension

### Class corresponding to 70 x 70 PatchGAN discriminator
### Let Ck denote a Convolution-BatchNorm-ReLU layer with k filters.
### Then, the architecutre for the discriminator is: 
### C64-C128-C256-C512

### The size of the image at each pass through layers of Discriminator: 

### z = N, 2C, H, W
### z1 = N, 64, H/2, W/2
### z2 = N, 128, H//4 -1 , W//4 - 1 
### z3 = N, 256, H//8 - 2, W //8 - 2
### z4 = N, 512, H//8 - 4, W//8 - 4
### z5 = N, 1, H//8 - 6, W//8 - 6 lose some size due to stride = 2  


### Why does this correspond to 70x70 patchGAN? Apparently, each pixel at output corresponds to some 70x70 patch of input 
### 

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, num_filters = [64, 128, 256, 512]):
        super().__init__()

        #First layer does not utilize batchnorm 
        #Takes as input two images concatenated along channel dimension, so in_channels*2 input channels

        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels*2, num_filters[0], kernel_size = 4, stride = 2, padding = 1, padding_mode = "reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        layers.append(self.initial_layer)
        in_channels = num_filters[0]

        for filter in num_filters[1:]:

            if filter  == num_filters[-1]:
                #Last layer utilizes stride of 1 
                layers.append(Conv_BN_Relu_Block(in_channels, filter, stride = 1))
            else:
                layers.append(Conv_BN_Relu_Block(in_channels, filter))


            in_channels = filter #update # of input channels when appending next layers

        ### After the last layer, a convolution is applied to map to
        ### a 1-dimensional output, followed by a Sigmoid function.

        layers.append( 
            
            nn.Sequential(
            nn.Conv2d( in_channels, out_channels = 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = "reflect"),
            nn.Sigmoid() #Map the output to be between 0,1 
            )
        
        )

        self.model = nn.Sequential(*layers) #Unpack list and create Discriminator architecture  

    def forward(self, x, y):
        x = torch.cat([x, y], dim = 1) #concatenate along channel dimension 
        return self.model(x) #and then the rest of the model. 


if __name__ == "__main__":
    N,C,H,W = (1,3,286, 286)
    x = torch.randn(N,C,H,W)
    y = torch.randn(N,C,H,W)
    model = Discriminator()
    logits = model(x,y)
    print(logits.shape) #too ez clown
    
