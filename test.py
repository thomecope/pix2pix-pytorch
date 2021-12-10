# torch packages:
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import torchvision.utils as utils

# python packages:
import time
import os

# our packages:
from Discriminator import Discriminator
from Generator import Generator
import config
import checkpoint
from dataset_final import Pix2pix_Dataset


def test(test_loader):

    # create instances of generator and discriminator
    generator = Generator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    discriminator = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)

    # put in training mode
    generator.train()
    discriminator.train()

    # define loss criterion
    bce = torch.nn.BCEWithLogitsLoss()
    l1 = torch.nn.L1Loss()

    # define optimizers
    opt_gen = torch.optim.Adam(generator.parameters(), lr = config.LR, betas = (config.BETA_1, config.BETA_2))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr = config.LR, betas = (config.BETA_1, config.BETA_2))

    # define scalers (automatic precision stuff)
    gen_scaler = amp.GradScaler()
    disc_scaler = amp.GradScaler()

    # restore checkpoint (if possible)
    print('Loading model...')
    
    # FOR ON GREAT LAKES
    # generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, start_epoch, stats = checkpoint.restore_checkpoint(
    #     generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, config.CKPT_PATH, cuda=True, force=config.FORCE)
    
    # FOR ON PC
    generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, start_epoch, stats = checkpoint.restore_checkpoint(
        generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, config.CKPT_PATH, cuda=False, force=config.FORCE)

    # put generator into eval mode
    generator.eval()
    with torch.no_grad(): 

        for idx, (inp, tar) in enumerate(test_loader):
            inp = inp.to(config.DEVICE)
            tar = tar.to(config.DEVICE)
            
            # use generator to make prediction
            pred = generator(inp)

            # saving image
            grid = utils.make_grid([torch.squeeze(inp), torch.squeeze(pred), torch.squeeze(tar)])*0.5+0.5
            
            path = os.path.join(config.TEST_SAVE_FOLDER)
            if not os.path.exists(path):
                os.makedirs(path)
                
            utils.save_image(grid, path + '/test-'+str(idx)+'.png')

    
    print('Finished Testing')
    


if __name__ == "__main__":

    testing_data = Pix2pix_Dataset(config.TEST_DATA_PATH, t_flag=False)
    test_loader = DataLoader(testing_data, batch_size=config.BATCH_SIZE,shuffle=False)
    test(test_loader)
    

