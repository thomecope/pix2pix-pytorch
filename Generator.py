import math
import torch
import torch.nn as nn
import torch.nn.functional as F

##Class to perform conv -> batch norm -> leaky relu. Effectively reduces spatial dimension of input 
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        ### Use nn.Sequential to group layers 

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias=False, padding_mode = "reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        ### Forward pass 
    def forward(self, x):
       return self.model(x)

       ### Up - convolutional layers used for accomodating up-sampling and skip connections as in traditional U-Net fashion
       ### Layers are structured as transpose_conv -> BN -> reLU -> dropout 

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout = True):
        super().__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())

        if dropout == True: 
            layers.append(nn.Dropout(p = 0.5))

        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
       return self.model(x)

       
        ### Now, create generator out of DownConv and UpConv blocks as well as initialization/bottleneck/output layers

        ### Let Ck denote a Convolution-BatchNorm-ReLU layer
        ### with k filters. CDk denotes a Convolution-BatchNorm-
        ### Dropout-ReLU layer with a dropout rate of 50%
        ### All convolutions are 4 × 4 spatial filters applied with stride 2. Convo-
        ### lutions in the encoder, and in the discriminator, downsample
        ### by a factor of 2, whereas in the decoder they upsample by a
        ### factor of 2.


        
class Generator(nn.Module):
    def __init__(self, in_channels = 3, num_filters = 64):
        super().__init__()
        
        ### The encoder-decoder architecture consists of:
        ### encoder:
        ### C64-C128-C256-C512-C512-C512-C512-C512
        
        
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size = 4, stride = 2, padding = 1, padding_mode = "reflect"),
            nn.LeakyReLU(0.2)

        )
        self.down1 = DownConv(num_filters, num_filters*2)
        self.down2 = DownConv(num_filters*2, num_filters*4)
        self.down3 = DownConv(num_filters*4, num_filters*8)
        self.down4 = DownConv(num_filters*8, num_filters*8)
        self.down5 = DownConv(num_filters*8, num_filters*8)
        self.down6 = DownConv(num_filters*8, num_filters*8)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters*8, num_filters*8, kernel_size = 4, stride = 2, padding = 1, padding_mode = "reflect"),
            nn.ReLU(),
        )

        ### The decoder architecure consists of:
        ### CD512-CD512-CD512-C512-C256-C128-C64
        ### We multiply the number of filters here because due to skip connections, 
        ### We concatenate layer i and n-i along channel dimension 

        self.up1 = UpConv(num_filters*8, num_filters*8, dropout = True )
        self.up2 = UpConv(num_filters*8*2, num_filters*8, dropout = True )
        self.up3 = UpConv(num_filters*8*2, num_filters*8, dropout = True )
        self.up4 = UpConv(num_filters*8*2, num_filters*8, dropout = False)
        self.up5 = UpConv(num_filters*8*2, num_filters*4, dropout = False )
        self.up6 = UpConv(num_filters*4*2, num_filters*2, dropout = False )
        self.up7 = UpConv(num_filters*2*2, num_filters, dropout = False )
        
        #Retain output size same size as input, and force to be non-negative by using tanh()

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(num_filters*2, in_channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh(),
        )

    def forward(self, x):

        d1 = self.init_layer(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        ###Concatenate activations at layer i and n-i along channel dimension 

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat( [up1, d7], dim = 1 ))
        up3 = self.up3(torch.cat( [up2, d6], dim = 1 ))
        up4 = self.up4(torch.cat( [up3, d5], dim = 1 ))
        up5 = self.up5(torch.cat( [up4, d4], dim = 1 ))
        up6 = self.up6(torch.cat( [up5, d3], dim = 1 ))
        up7 = self.up7(torch.cat( [up6, d2], dim = 1 ))
        return self.output_layer( torch.cat([up7, d1], dim = 1))


if __name__ == "__main__":
    N,C,H,W = (1, 3, 256, 256)
    x = torch.randn((N,C,H,W))
    model = Generator(in_channels = C)
    preds = model(x)
    print(preds.shape)
    if preds.shape != (N,C,H,W):
        print("Size error somewhere!!")
    else:
        print("Input and output sizes match!")
    