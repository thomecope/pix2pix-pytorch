import torch 

# optimization params
LR = 0.0002
BETA_1 = 0.5
BETA_2 = 0.999

# training info
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
EPOCHS = 20
BATCH_SIZE = 1

# hyperparams
LAMBDA = 100

# load checkpoint? 
FORCE = True
# CKPT_PATH = 'checkpoints/pix2pix/facades'
CKPT_PATH = 'checkpoints/pix2pix/cells'
# CKPT_PATH = 'checkpoints/pix2pix/cityscapes'
# CKPT_PATH = 'checkpoints/pix2pix/brain'
# CKPT_PATH = 'checkpoints/pix2pix/dogs'
# CKPT_PATH = 'checkpoints/pix2pix/households'
# CKPT_PATH = 'checkpoints/pix2pix/cityscapes'
MAX_TO_KEEP = 3

# saving images
'''SAVE FOLDER: SAMPLE IMAGES
   TEST SAVE FOLDER: TEST IMAGES'''

# SAVE_FOLDER = 'sample_images/facades_eric'
SAVE_FOLDER = 'sample_images/cells'
# SAVE_FOLDER = 'sample_images/cityscapes'
# SAVE_FOLDER = 'sample_images/brain'
# SAVE_FOLDER = 'sample_images/dogs'
# TEST_SAVE_FOLDER = 'test_images/households'
# TEST_SAVE_FOLDER = 'test_images/facades_eric_sorted'
TEST_SAVE_FOLDER = 'test_images/cells'
# TEST_SAVE_FOLDER = 'test_images/cityscapes'
# TEST_SAVE_FOLDER = 'test_images/brain'
# TEST_SAVE_FOLDER = 'test_images/dogs'

# dataset information
# IN_CHANNELS = 3
IN_CHANNELS = 1
GRAYSCALE = True
# TRAIN_DATA_PATH = 'facades/train'
# VAL_DATA_PATH = 'facades/val'
TRAIN_DATA_PATH = 'cell_data/train'
TEST_DATA_PATH = 'cell_data/test'
TEST_DATA_PATH = 'cell_data/test'
# TRAIN_DATA_PATH = 'cityscapes/train'
# VAL_DATA_PATH = 'cityscapes/val'
# TEST_DATA_PATH = 'cityscapes/val'
# TRAIN_DATA_PATH = 'brain_data/train'
# VAL_DATA_PATH = 'brain_data/val'
# TEST_DATA_PATH = 'brain_data/val'
# TRAIN_DATA_PATH = 'dog_data/train'
# VAL_DATA_PATH = 'dog_data/test'
# TEST_DATA_PATH = 'dog_data/test'
# TEST_DATA_PATH = 'cityscapes/val'

# import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TRAIN_DIR = "data/train"
# VAL_DIR = "data/val"
# LEARNING_RATE = 2e-4
# BATCH_SIZE = 16
# NUM_WORKERS = 2
# IMAGE_SIZE = 256
# CHANNELS_IMG = 3
# L1_LAMBDA = 100
# LAMBDA_GP = 10
# NUM_EPOCHS = 500
# LOAD_MODEL = False
# SAVE_MODEL = False
# CHECKPOINT_DISC = "disc.pth.tar"
# CHECKPOINT_GEN = "gen.pth.tar"

# both_transform = A.Compose(
#     [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
# )

# transform_only_input = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),
#         #A.ColorJitter(p=0.2),
#         A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         ToTensorV2(),
#     ]
# )

# transform_only_mask = A.Compose(
#     [
#         A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         ToTensorV2(),
#     ]
# )


