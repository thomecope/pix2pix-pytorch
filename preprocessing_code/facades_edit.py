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
import random


###############################################################################################


import cv2
import numpy as np
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import shutil

all_images = []
mask_images = []
facade_images = []
concatenated_ims = []


filepathroot1 = 'input_images'
filepathroot2 = 'segment_rgb'

images = [f for f in sorted(listdir(filepathroot1)) if isfile(join(filepathroot1, f))]

for image in images:
    
    #full_im_path = filepathroot1 + '/' + image
    facade_images.append(image)

segs = [f for f in sorted(listdir(filepathroot2)) if isfile(join(filepathroot2, f))]

for image in segs:
    
    #full_im_path = filepathroot2 + '/' + image
    entry_new = image.replace('.png','.jpg')

    mask_images.append(entry_new)

        
for image in facade_images:
    
    #print(image)
    
    if image in mask_images:
        
        if image != 'cmp_b0250.jpg':
    
            fullpath_image = filepathroot1 + '/' + image
            entry_new = image.replace('.jpg','.png')

            fullpath_mask = filepathroot2 + '/'  + entry_new

            im_A = cv2.imread(fullpath_image,1)

            imA2 = cv2.resize(im_A, dsize=(256, 256))
            imA3 = cv2.cvtColor(imA2, cv2.COLOR_BGR2RGB)

            im_B = cv2.imread(fullpath_mask,1)    
            imB2 = cv2.resize(im_B, dsize=(256, 256))
            imB3 = cv2.cvtColor(imB2, cv2.COLOR_BGR2RGB)
            
            im_AB = np.concatenate([imA3, imB3], 1)                                                
            concatenated_ims.append(im_AB)  
            
#     else:
#         print(image)
        

if image in mask_images:
    
    if image not in facade_images:
        
        print(image)
            


###############################################################################################

for idx, entry in enumerate(concatenated_ims):
    filename = "fcn_ims_concat_sorted_colors/" + str(idx) + '.jpg'
    im = Image.fromarray(entry)
    im.save(filename)
