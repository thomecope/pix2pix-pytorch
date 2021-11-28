import torch
import torch.cuda.amp as amp
from Discriminator import Discriminator
from Generator import Generator
import config
import checkpoint

def _disc_loss(disc_real, disc_fake,  bce):

    # discriminator losses
    real_loss = bce(disc_real, torch.ones_like(disc_real))
    fake_loss = bce(disc_fake, torch.zeros_like(disc_fake))

    return (real_loss + fake_loss)/2

def _gen_loss(disc_fake, pred, tar, bce, l1):

    # discriminator losses
    gan_loss = bce(disc_fake, torch.ones_like(disc_fake))
    l1_loss = l1(pred, tar)

    return gan_loss + config.LAMBDA * l1_loss 

def _train_epoch(train_loader, generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, bce, l1):
    for idx, (inp, tar) in enumerate(train_loader):
        
        # clear gradients:
        opt_disc.zero_grad()
        opt_gen.zero_grad()

        # move to variables to cuda
        inp = inp.cuda()
        tar = tar.cuda()

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

# def _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, epoch):
#     """
#     Evaluates the model on the train and validation set.
#     """
#     stat = []
#     for data_loader in [val_loader, train_loader]:
#         y_true, y_pred, running_loss = evaluate_loop(data_loader, model, criterion)
#         total_loss = np.sum(running_loss) / y_true.size(0)
#         total_acc = accuracy(y_true, y_pred)
#         stat += [total_acc, total_loss]
#     plotter.stats.append(stat)
#     plotter.log_cnn_training(epoch)
#     plotter.update_cnn_training_plot(epoch)


# def evaluate_loop(data_loader, model, criterion=None):
#     model.eval()
#     y_true, y_pred, running_loss = [], [], []
#     for X, y in data_loader:
#         with torch.no_grad():
#             output = model(X)
#             predicted = predictions(output.data)
#             y_true.append(y)
#             y_pred.append(predicted)
#             if criterion is not None:
#                 running_loss.append(criterion(output, y).item() * X.size(0))
#     model.train()
#     y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
#     return y_true, y_pred, running_loss


def train(generator, discriminator, train_loader):
    """
    generator: model of gen
    discriminator: model of disc
    """

    # define loss criterion
    bce = torch.nn.BCELoss()
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

    # Create plotter
    # plot_name = config['plot_name'] if 'plot_name' in config else 'CNN'
    # plotter = Plotter(stats, plot_name)

    # Evaluate the model
    # _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, start_epoch)

    # iterate through epochs
    for epoch in range(start_epoch, config.EPOCHS):
        # train model
        _train_epoch(train_loader, generator, discriminator, opt_gen, opt_disc, bce, l1)

        # Evaluate model on training and validation set
        # _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, epoch + 1)

        # save checkpoint (saving none for stats currently)

        checkpoint.save_checkpoint(
            generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, 
            epoch + 1, config.CKPT_PATH, None)
        print('Epoch ', epoch, ' out of ', config.EPOCHS, ' complete')
    
    print('Finished Training')

    # Save figure and keep plot open
    # plotter.save_cnn_training_plot()
    # plotter.hold_training_plot()

def main(dataset):
    # create instances of generator and discriminator
    generator = Generator(in_channels=dataset.in_channels).cuda()
    discriminator = Discriminator(in_channels=dataset.in_channels).cuda()

    # put in training mode
    generator.train()
    discriminator.train()

    train(generator, discriminator, dataset)


if __name__ == "__main__":
    
    # declare dataset
    dataset = Cityscape() 
    main(dataset)

