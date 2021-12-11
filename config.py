import torch 

# optimization params per the paper
LR = 0.0002
BETA_1 = 0.5
BETA_2 = 0.999

# hyperparams
LAMBDA = 100

# training info
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
EPOCHS = 200
BATCH_SIZE = 1

# load checkpoint? 
FORCE = True
CKPT_PATH = 'checkpoints/pix2pix/facades'
MAX_TO_KEEP = 3

# paths for saving images
SAMPLE_SAVE_FOLDER = 'sample_images/facades'
TEST_SAVE_FOLDER = 'test_images/facades'


# dataset information
# IN_CHANNELS = 1
# GRAYSCALE = True
IN_CHANNELS = 3
TRAIN_DATA_PATH = 'facades/train'
VAL_DATA_PATH = 'facades/val'
TEST_DATA_PATH = 'facades/test'