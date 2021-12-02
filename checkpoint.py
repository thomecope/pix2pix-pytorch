# EECS 545 Fall 2021 -- modified for pix2pix
import itertools
import os
import torch
import config


def save_checkpoint(generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, epoch, checkpoint_dir, stats):
    """
    Save model checkpoint.
    """
    state = {
        'epoch': epoch,
        'm1_state_dict': generator.state_dict(),
        'm2_state_dict': discriminator.state_dict(),
        'opt1_state_dict': opt_gen.state_dict(),
        'opt2_state_dict': opt_disc.state_dict(),
        's1_state_dict': gen_scaler.state_dict(),
        's2_state_dict': disc_scaler.state_dict(),
        'stats': stats
    }

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filename = os.path.join(checkpoint_dir,'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)

    delete_old_checkpoint(checkpoint_dir)



def restore_checkpoint(generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, checkpoint_dir, cuda=False, force=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model, the current epoch, and training losses.
    """
    def get_epoch(cp):
        return int(cp.split('epoch=')[-1].split('.checkpoint.pth.tar')[0])

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir) if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]
    cp_files.sort(key=lambda x: get_epoch(x))

    # No saved or don't want it: 
    if not cp_files or not force:    
        return generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, 0, []
    # Load the most recent:
    else:
        epochs = [get_epoch(cp) for cp in cp_files]
        inp_epoch = max(epochs)
        filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(inp_epoch))
        print("Loading from checkpoint {}".format(filename))
        if cuda:
            checkpoint = torch.load(filename)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

        try:
            stats = checkpoint['stats']
            generator.load_state_dict(checkpoint['m1_state_dict'])
            discriminator.load_state_dict(checkpoint['m2_state_dict'])
            opt_gen.load_state_dict(checkpoint['opt1_state_dict'])
            opt_disc.load_state_dict(checkpoint['opt2_state_dict'])
            gen_scaler.load_state_dict(checkpoint['s1_state_dict'])
            disc_scaler.load_state_dict(checkpoint['s2_state_dict'])
            print("=> Successfully restored checkpoint (trained for {} epochs)".format(checkpoint['epoch']))
        except:
            print("=> Checkpoint not successfully restored")
            raise

        return generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, inp_epoch, stats

    # if not cp_files:
    #     print('No saved model parameters found')
    #     if force:
    #         raise Exception('Checkpoint not found')
    #     else:
    #         return generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, 0, []

    # # Find latest epoch
    # epochs = [get_epoch(cp) for cp in cp_files]

    # if not force:
    #     epochs = [0] + epochs
    #     print('Which epoch to load from? Choose from epochs below:')
    #     print(epochs)
    #     print('Enter 0 to train from scratch.')
    #     print(">> ", end='')
    #     inp_epoch = int(input())
    #     if inp_epoch not in epochs:
    #         raise Exception("Invalid epoch number")
    #     if inp_epoch == 0:
    #         print("Checkpoint not loaded")
    #         clear_checkpoint(checkpoint_dir)
    #         return generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, 0, []
    # else:
    #     print('Which epoch to load from? Choose from epochs below:')
    #     print(epochs)
    #     print(">> ", end='')
    #     inp_epoch = int(input())
    #     if inp_epoch not in epochs:
    #         raise Exception("Invalid epoch number")

    # filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    # print("Loading from checkpoint {}".format(filename))

    # if cuda:
    #     checkpoint = torch.load(filename)
    # else:
    #     # Load GPU model on CPU
    #     checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    # try:
    #     stats = checkpoint['stats']
    #     generator.load_state_dict(checkpoint['m1_state_dict'])
    #     discriminator.load_state_dict(checkpoint['m2_state_dict'])
    #     opt_gen.load_state_dict(checkpoint['opt1_state_dict'])
    #     opt_disc.load_state_dict(checkpoint['opt2_state_dict'])
    #     gen_scaler.load_state_dict(checkpoint['s1_state_dict'])
    #     disc_scaler.load_state_dict(checkpoint['s2_state_dict'])
    #     print("=> Successfully restored checkpoint (trained for {} epochs)".format(checkpoint['epoch']))
    # except:
    #     print("=> Checkpoint not successfully restored")
    #     raise

    # return generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):
    """
    Delete all checkpoints in directory.
    """
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")

def delete_old_checkpoint(checkpoint_dir):
    """
    Delete all checkpoints in directory.
    """
    def get_epoch(cp):
        return int(cp.split('epoch=')[-1].split('.checkpoint.pth.tar')[0])

    filelist = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch=') and f.endswith('.checkpoint.pth.tar')]
    filelist.sort(key=lambda x: get_epoch(x))
    
    epochs = [get_epoch(f) for f in filelist]
    lowest = min(epochs)
    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(lowest))

    if len(filelist)>config.MAX_TO_KEEP:
        os.remove(os.path.join(checkpoint_dir, filename)
