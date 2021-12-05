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
from dataset_temp import Pix2pix_Dataset


def _disc_loss(disc_real, disc_fake,  bce):

    # discriminator losses
    real_loss = bce(disc_real, torch.ones_like(disc_real))
    fake_loss = bce(disc_fake, torch.zeros_like(disc_fake))

    return (real_loss + fake_loss)/2

def _gen_loss(disc_fake, pred, tar, bce, l1):

    # generator losses
    gan_loss = bce(disc_fake, torch.ones_like(disc_fake))
    l1_loss = l1(pred, tar)

    return gan_loss + config.LAMBDA * l1_loss 

def _train_epoch(train_loader, generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, bce, l1, epoch):
    
    for idx, (inp, tar) in enumerate(train_loader):
        
        # clear gradients:
        opt_disc.zero_grad()
        opt_gen.zero_grad()

        # move to variables to cuda
        inp = inp.to(config.DEVICE)
        tar = tar.to(config.DEVICE)

        # calc values with fp16
        with amp.autocast():
            # evaluate models
            pred = generator(inp)
            disc_real = discriminator(inp, tar)
            disc_fake = discriminator(inp, pred.detach()) # detach to retain computational graph

            # calculate discriminator loss first
            disc_loss = _disc_loss(disc_real, disc_fake, bce)

        # update scaler (scales losses to avoid underflow)
        # take backward step on discriminator
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(opt_disc)
        disc_scaler.update()

        # perform all steps again with new discriminator for updating generator
        with amp.autocast():
            disc_fake = discriminator(inp, pred)
            gen_loss = _gen_loss(disc_fake, pred, tar, bce, l1)

        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(opt_gen)
        gen_scaler.update()

        # if epoch%5==0:
        #     print('Disc loss: ', disc_loss)
        #     print('Gen loss: ', gen_loss)

#

def evaluate_epoch(data_loader, generator, epoch):
    inp, tar = next(iter(data_loader))
    inp = inp.to(config.DEVICE)
    tar = tar.to(config.DEVICE)
    
    generator.eval()
    with torch.no_grad():
        pred = generator(inp)
        grid = utils.make_grid([torch.squeeze(inp), torch.squeeze(pred), torch.squeeze(tar)])*0.5+0.5
        
        path = os.path.join(config.SAVE_FOLDER)
        if not os.path.exists(path):
            os.makedirs(path)
            
        utils.save_image(grid, path + '/test'+str(epoch)+'.png')

    generator.train()


def train(train_loader, val_loader):
    """
    generator: model of gen
    discriminator: model of disc
    """

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
    generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, start_epoch, stats = checkpoint.restore_checkpoint(
        generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, config.CKPT_PATH, cuda=True, force=config.FORCE)

    # iterate through epochs
    for epoch in range(start_epoch, config.EPOCHS):
        # train model
        print('===========================================================')
        print('Training Epoch ', epoch, '...')
        start_time = time.time()
        _train_epoch(train_loader, generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, bce, l1, epoch)
        
        if epoch%5 == 0:
            evaluate_epoch(val_loader, generator, epoch)
             
        # save checkpoint (saving none for stats currently)
        checkpoint.save_checkpoint(
            generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, 
            epoch + 1, config.CKPT_PATH, None)
        print('Epoch ', epoch+1, ' out of ', config.EPOCHS, ' complete in ', time.time()-start_time)
    
    print('Finished Training')


if __name__ == "__main__":

    training_data = Pix2pix_Dataset(config.TRAIN_DATA_PATH, t_flag=True)
    train_loader = DataLoader(training_data, batch_size=config.BATCH_SIZE,shuffle=True)

    validation_data = Pix2pix_Dataset(config.VAL_DATA_PATH, t_flag=False)
    validation_loader = DataLoader(validation_data, batch_size=config.BATCH_SIZE,shuffle=False)
    
    # declare dataset
    train(train_loader, validation_loader)


    

