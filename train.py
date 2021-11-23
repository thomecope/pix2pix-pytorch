import torch
from Discriminator import Discriminator
from Generator import Generator
import config
import checkpoint

def _disc_loss(disc_real, disc_fake,  bce):

    # discriminator losses
    real_loss = bce(disc_real, torch.ones_like(disc_real))
    fake_loss = bce(disc_fake, torch.zeros_like(disc_fake))

    return real_loss + fake_loss

def _gen_loss(disc_fake, pred, tar, bce, l1):

    # discriminator losses
    gan_loss = bce(disc_fake, torch.ones_like(disc_real))
    l1_loss = l1(pred, tar)

    return gan_loss + config.LAMBDA * l1_loss

    

def _train_epoch(train_loader, generator, discriminator, opt_gen, opt_disc, bce, l1):
    for idx, (inp, tar) in enumerate(train_loader):

        # evaluate models
        pred = generator(inp)
        disc_real = discriminator(inp, tar)
        disc_fake = discriminator(inp, pred.detach()) # detach to retain computational graph

        # update discriminator
        opt_disc.zero_grad()
        disc_loss = _disc_loss(disc_real, disc_fake, bce)
        disc_loss.backward()
        opt_disc.step()

        # recalculate with new discriminator
        disc_fake = discriminator(inp, pred)

        #update generator
        opt_gen.zero_grad()
        gen_loss = _gen_loss(disc_fake, pred, tar, bce, l1)
        gen_loss.backward()
        opt_gen.step()

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


def _train(generator, discriminator):
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

    # restore checkpoint (if possible)
    print('Loading model...')
    force = config.FORCE
    model, start_epoch, stats = checkpoint.restore_checkpoint(model, config['ckpt_path'], force=force)

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

        # save checkpoint
        # checkpoint.save_checkpoint(model, epoch + 1, config['ckpt_path'], plotter.stats)
        checkpoint.save_checkpoint(model, epoch + 1, config['ckpt_path'], None)
        print('Epoch ', epoch, ' out of ', config.EPOCHS, ' complete')
    
    print('Finished Training')

    # Save figure and keep plot open
    # plotter.save_cnn_training_plot()
    # plotter.hold_training_plot()

def _main(dataset):
    generator = Generator(in_channels=dataset.in_channels)
    discriminator = Discriminator(in_channels=in_channels)
    _train(generator, discriminator, dataset)


if __name__ == "__main__":
    
    # declare dataset
    dataset = None 
    _main(dataset)

