# optimization params
LR = 0.0002
BETA_1 = 0.5
BETA_2 = 0.999

# training info
dev = "cuda" if torch.cuda.is_available() else dev = "cpu"
DEVICE = torch.device(dev)  
EPOCHS = 20
BATCH_SIZE = 1

# hyperparams
LAMBDA = 100

# load checkpoint? 
FORCE = False
CKPT_PATH = 'checkpoints/pix2pix'
MAX_TO_KEEP = 5

# dataset information
IN_CHANNELS = 3
TRAIN_DATA_PATH = 'facades/train'
VAL_DATA_PATH = 'facades/val'