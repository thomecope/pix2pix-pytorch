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


def _disc_loss(disc_tar, disc_pred,  bce):
    """
    inputs:
    disc_tar - the output of the discriminator given the real image
    disc_pred - the output of the discriminator given the predicted image
    bce - the loss criterion, BCEwithlogitsloss()

    outputs:
    the computed loss for the discriminator
    """

    # discriminator losses: 
    tar_loss = bce(disc_tar, torch.ones_like(disc_tar)) 
    pred_loss = bce(disc_pred, torch.zeros_like(disc_pred))

    return (tar_loss + pred_loss)/2

def _gen_loss(disc_pred, pred, tar, bce, l1):
    """
    inputs:
    disc_pred - the output of the discriminator given the predicted image
    pred - the predicted image
    tar - the target image
    bce - loss criterion, BCEWithLogitsLoss()
    l1 - loss criterion, L1Loss

    outputs:
    the computed loss for the generator
    """

    # generator losses:
    gan_loss = bce(disc_pred, torch.ones_like(disc_pred))
    l1_loss = l1(pred, tar)

    return gan_loss + config.LAMBDA * l1_loss 

def _train_epoch(train_loader, generator, discriminator, opt_gen, opt_disc, bce, l1, epoch):
    """
    Given the above information, trains the models on the dataset
    Running this once trains the models for one epoch
    """
    
    for inp, tar in train_loader:
        
        # clear gradients:
        opt_disc.zero_grad()
        opt_gen.zero_grad()

        # move to variables to cuda
        inp = inp.to(config.DEVICE)
        tar = tar.to(config.DEVICE)

        # evaluate models
        pred = generator(inp)
        disc_tar = discriminator(inp, tar)
        disc_pred = discriminator(inp, pred.detach()) # detach to retain computational graph

        # calculate discriminator loss first
        disc_loss = _disc_loss(disc_tar, disc_pred, bce)

        # take backward step on discriminator
        disc_loss.backward()
        opt_disc.step()

        # perform all steps again with new discriminator for updating generator
        disc_pred = discriminator(inp, pred)
        gen_loss = _gen_loss(disc_pred, pred, tar, bce, l1)
        
        gen_loss.backward()
        opt_gen.step()

def evaluate_epoch(data_loader, generator, epoch):
    """
    Generates one image every time run to give understanding of training progression

    inputs:
    data_loader - dataloader containing validation dataset
    generator - model for generator
    epoch - current epoch number
    """

    # take first image from data_loader
    inp, tar = next(iter(data_loader))
    inp = inp.to(config.DEVICE)
    tar = tar.to(config.DEVICE)
    
    # put generator into eval mode
    generator.eval()
    with torch.no_grad():

        # use generator to make prediction
        pred = generator(inp)

        # saving image
        grid = utils.make_grid([torch.squeeze(inp,0), torch.squeeze(pred,0), torch.squeeze(tar,0)])*0.5+0.5
        
        path = os.path.join(config.SAMPLE_SAVE_FOLDER)
        if not os.path.exists(path):
            os.makedirs(path)
            
        utils.save_image(grid, path + '/sample-'+str(epoch)+'.png')

    # put generator back into training mode
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

    # define loss criteria
    bce = torch.nn.BCEWithLogitsLoss()
    l1 = torch.nn.L1Loss()

    # define optimizers
    opt_gen = torch.optim.Adam(generator.parameters(), lr = config.LR, betas = (config.BETA_1, config.BETA_2))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr = config.LR, betas = (config.BETA_1, config.BETA_2))

    # restore checkpoint (if possible)
    print('Loading model...')
    generator, discriminator, opt_gen, opt_disc, start_epoch, stats = checkpoint.restore_checkpoint(
        generator, discriminator, opt_gen, opt_disc, config.CKPT_PATH, cuda=True, force=config.FORCE)

    # iterate through epochs
    for epoch in range(start_epoch, config.EPOCHS):
        
        # train model
        print('===========================================================')
        print('Training Epoch ', epoch, '...')
        start_time = time.time()
        _train_epoch(train_loader, generator, discriminator, opt_gen, opt_disc, bce, l1, epoch)
        
        # generate sample image every 5
        if epoch%5 == 0:
            evaluate_epoch(val_loader, generator, epoch)
             
        # save checkpoint
        checkpoint.save_checkpoint(
            generator, discriminator, opt_gen, opt_disc, epoch + 1, config.CKPT_PATH, None)
        print('Epoch ', epoch+1, ' out of ', config.EPOCHS, ' complete in ', time.time()-start_time)
    
    print('Finished Training')


if __name__ == "__main__":

    # create data loaders (validation used for evaluating epochs)
    training_data = Pix2pix_Dataset(config.TRAIN_DATA_PATH, t_flag=True, grayscale=config.GRAYSCALE)
    train_loader = DataLoader(training_data, batch_size=config.BATCH_SIZE,shuffle=True)

    validation_data = Pix2pix_Dataset(config.VAL_DATA_PATH, t_flag=False, grayscale=config.GRAYSCALE)
    validation_loader = DataLoader(validation_data, batch_size=config.BATCH_SIZE,shuffle=False)
    
    # train network on dataset
    train(train_loader, validation_loader)


    

