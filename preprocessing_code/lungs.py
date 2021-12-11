# preprocessing code was developed using Jupyter notebooks. it probably will not run here because many snippets of
# code were added together so as to not have so many files. that was assumed to be okay since it is for the 
# preprocessing.

import numpy as np
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import shutil
from pdb import set_trace as st
import cv2
import argparse

import cv2
import numpy as np
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import shutil


all_images = []
mask_images = []
mri_images = []

filepathroot_images = 'eecs545project/lung_images/images/images/'
filepathroot_masks = 'eecs545project/lung_images/masks/masks/'


images = [f for f in listdir(filepathroot_images) if isfile(join(filepathroot_images, f))]
masks = [f for f in listdir(filepathroot_masks) if isfile(join(filepathroot_masks, f))]

titles_strip_mask = {}

for entry in masks:
    
    entry_new = entry.replace('_mask','')
    titles_strip_mask[entry_new] = entry
    
concatenated_ims = []
    
for image in images:
    
    if image in titles_strip_mask:
    
        fullpath_image = filepathroot_images + image
        fullpath_mask = filepathroot_masks  + titles_strip_mask[image]
    
        im_A = cv2.imread(fullpath_image,1)
        im_B = cv2.imread(fullpath_mask,1)
        im_AB = np.concatenate([im_A, im_B], 1)
                                                                    
        concatenated_ims.append(im_AB)
                                                                    
        

for idx, entry in enumerate(concatenated_ims):
    filename = "lung_ims_concat/" + str(idx) + '.jpg'
    im = Image.fromarray(entry)
    im.save(filename)
