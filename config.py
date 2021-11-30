# # # optimization params
# # LR = 0.0002
# # BETA_1 = 0.5
# # BETA_2 = 0.999

# # # training info
# # EPOCHS = None

# # # hyperparams
# # LAMBDA = 100

# # # load checkpoint? 
# # FORCE = False
# # CKPT_PATH = 'checkpoints/pix2pix'
# # MAX_TO_KEEP = 5



# import torch
# import torchvision.transforms as transforms
# #import albumentations as A
# #from albumentations.pytorch import ToTensorV2

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

# both_transform = transforms.Compose(
#     transforms.Resize([256,256])
# )

# transform_only_input = transforms.Compose(
#     [
#         #transforms.HorizontalFlip(p=0.5),
#         #transforms.ColorJitter(p=0.2),
#         #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         transforms.ToTensor(),
#     ]
# )

# transform_only_mask = transforms.Compose(
#     [
#         #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         transforms.ToTensor(),
#     ]
# )

# # def get_transforms(self):
# #         transform_list = [
# #             #transforms.Resize([32,32]), 
# #             #transforms.CenterCrop(299),
# #             transforms.ToTensor(),
# #         ]
# #         transform = transforms.Compose(transform_list)
# #         return transform


import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        #A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)