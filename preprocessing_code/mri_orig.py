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

all_images = []
mask_images = []
mri_images = []

filepathroot = 'eecs545_dataset/lgg-mri-segmentation/kaggle_3m'
folders = [name for name in os.listdir(filepathroot) if os.path.isdir(os.path.join(filepathroot, name))]

for subject in folders:

    impathroot = filepathroot + '/' + subject
    images = [f for f in listdir(impathroot) if isfile(join(impathroot, f))]
    
    for image in images:
    
        full_im_path = impathroot + '/' + image
        all_images.append(full_im_path)
        

for entry in all_images:
        
    if 'mask' in entry:
        #entry_new = entry.replace('_mask','')
        mask_images.append(entry)
    else:
        mri_images.append(entry)
      
    
# mask_ims_path = 'path/to/data/A'
# mri_ims_path = 'path/to/data/B'
        
        
# for image in mask_images:
#     shutil.move(image, mri_ims_path)
    

# for image in mri_images:
#     shutil.move(image, mri_ims_path)

    

# completeName = os.path.join(save_path, file_name)
# print(completeName)
# OUTPUT
# /home/test.txt

# file1 = open(completeName, "w")
# file1.write("file information")
# file1.close()




im = Image.open('eecs545_dataset/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_11_mask.tif')

im2 = Image.open('eecs545_dataset/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_11.tif')

#'TCGA_<institution-code>_<patient-id>_<slice-number>.tif'
#im.show()

im

image = np.array(im)
image2 = np.array(im2)
type(image)
image.shape

#im = Image.open(all_images[4839])

# save_path = '/home'
# file_name = "test.txt"

# completeName = os.path.join(save_path, file_name)
# print(completeName)
# OUTPUT
# /home/test.txt

# file1 = open(completeName, "w")
# file1.write("file information")
# file1.close()

import cv2

im_A = cv2.imread('eecs545_dataset/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_11_mask.tif',1)
im_B = cv2.imread('eecs545_dataset/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_11.tif',1)


im_A.shape

for i in range(256):
    for j in range(256):
        
        if(im_A[i,j,1]) != 0:
            print((im_A[i,j,0]))

#im_AB = np.concatenate([im_A, im_B], 1)



impathroot = 'B'
images = [f for f in listdir(impathroot) if isfile(join(impathroot, f))]

titles_strip_mask = []

for entry in images:
    
    entry_new = entry.replace('_mask','')
    titles_strip_mask.append(entry_new) 
    
    
    

paths = (os.path.join(root, filename)
        for root, _, filenames in os.walk('path/to/data/B')
        for filename in filenames)

for path in paths:
    # the '#' in the example below will be replaced by the '-' in the filenames in the directory
    newname = path.replace('_mask', '')
    if newname != path:
        os.rename(path, newname)


mri_paths = 'path/to/data/A/'
mask_paths = 'path/to/data/B/'


mri_images = [f for f in listdir(mri_paths) if isfile(join(mri_paths, f))]
mask_images = [f for f in listdir(mask_paths) if isfile(join(mask_paths, f))]

concatenated_ims = []

for image in mri_images:
    
    fullpathA = mri_paths + image
    fullpathB = mask_paths + image
    im_A = cv2.imread(fullpathA,1)
    im_B = cv2.imread(fullpathB,1)
    im_AB = np.concatenate([im_A, im_B], 1)
    
    concatenated_ims.append(im_AB)
    

for idx, entry in enumerate(concatenated_ims):
    filename = "brain_images_concatenated/" + str(idx) + '.jpg'
    im = Image.fromarray(entry)
    im.save(filename)
    

save_path = 'path/to/data/A/'
completeName = os.path.join(save_path, file_name)
print(completeName)
OUTPUT
/home/test.txt

file1 = open(completeName, "w")
file1.write("file information")
file1.close()


# new stuff as of 11/24/21


im = concatenated_ims[1999]
imL = im[:,0:256,:]
imR = im[:,256:512,:]
Image.fromarray(imL)
imR.shape


start_idx_col = random.randrange(0, 29)
end_idx_col = start_idx_col + 256
start_idx_row = random.randrange(0, 29)
end_idx_row = start_idx_row + 256

large_im_L = cv2.resize(imL, dsize=(286, 286))
small_im_L = large_im_L[start_idx_row:end_idx_row,start_idx_col:end_idx_col]

large_im_R = cv2.resize(imR,dsize=(286, 286))
small_im_R = large_im_R[start_idx_row:end_idx_row,start_idx_col:end_idx_col]

Image.fromarray(small_im_R)


# Thanksgiving
import cv2
import random

concatenated_images_randcrop = []

for im in concatenated_ims:
    imL = im[:,0:256,:]
    imR = im[:,256:512,:]    
    
    start_idx_col = random.randrange(0, 29)
    start_idx_row = random.randrange(0, 29)

    end_idx_col = start_idx_col + 256
    end_idx_row = start_idx_row + 256

    large_im_L = cv2.resize(imL, dsize=(286, 286))
    large_im_R = cv2.resize(imR,dsize=(286, 286))

    small_im_L = large_im_L[start_idx_row:end_idx_row,start_idx_col:end_idx_col]
    small_im_R = large_im_R[start_idx_row:end_idx_row,start_idx_col:end_idx_col]
    
    im_small_both = np.concatenate([small_im_L,small_im_R],1)
        
    concatenated_images_randcrop.append(im_small_both)
    

for idx, entry in enumerate(concatenated_images_randcrop):
    filename = "brain_images_concatenated_randcrop/" + str(idx) + '.jpg'
    im = Image.fromarray(entry)
    im.save(filename)



# Thanksgiving
import cv2
import random

concatenated_images_randflip = []

for im in concatenated_images_randcrop:
    
    vert_flip = random.randrange(0, 2)
    horiz_flip = random.randrange(0, 2)
    
    if vert_flip == 1:
        
        ima = Image.fromarray(im)
        out = ima.transpose(Image.FLIP_TOP_BOTTOM)
        image_flipped = np.array(out)
        
    if horiz_flip == 1:
        
        imL = Image.fromarray(image_flipped[:,0:256,:])
        imR = Image.fromarray(image_flipped[:,256:512,:])
        outL = imL.transpose(Image.FLIP_LEFT_RIGHT)
        outR = imR.transpose(Image.FLIP_LEFT_RIGHT)
        concatenated = np.concatenate([outL,outR],1)
        
    concatenated_images_randflip.append(concatenated)
    

for idx, entry in enumerate(concatenated_images_randflip):
    filename = "brain_images_concatenated_randflip/" + str(idx) + '.jpg'
    im = Image.fromarray(entry)
    im.save(filename)


    all_images = []
mask_images = []
mri_images = []

filepathroot = 'eecs545_dataset/kaggle_3m'
folders = [name for name in os.listdir(filepathroot) if os.path.isdir(os.path.join(filepathroot, name))]

for subject in folders:

    impathroot = filepathroot + '/' + subject
    images = [f for f in listdir(impathroot) if isfile(join(impathroot, f))]
    
    for image in images:
    
        full_im_path = impathroot + '/' + image
        all_images.append(full_im_path)
        