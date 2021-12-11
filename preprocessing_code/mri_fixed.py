# preprocessing code was developed using Jupyter notebooks. it probably will not run here because many snippets of
# code were added together so as to not have so many files. that was assumed to be okay since it is for the 
# preprocessing.

import numpy as np
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import shutil
import cv2

images_fixed = []


impathroot = 'brain_ims_small/test'
ims = [f for f in listdir(impathroot) if isfile(join(impathroot, f))]


image = cv2.imread('brain_ims_small/test/' + entry,1)
    

for entry in ims:
    
    image = cv2.imread('brain_ims_small/test/' + entry,1)
    mask = image[:,256:512]
    brain = image[:,0:256]

    for i in range(256):
        for j in range(256):

            if mask[i,j,0] == 0:
                mask[i,j,0] = 0
                mask[i,j,1] = 255
                mask[i,j,2] = 0
            else:
                mask[i,j,0] = 255
                mask[i,j,1] = 0
                mask[i,j,2] = 0
    
    images_fixed.append(np.concatenate([brain, mask], 1))


    
    
for idx, entry in enumerate(images_fixed):
    filename = "brain_ims_small_fixed/test/" + str(idx) + '.jpg'
    im = Image.fromarray(entry)
    im.save(filename)