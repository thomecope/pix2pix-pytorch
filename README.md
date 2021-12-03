# Image-To-Image Translation
PyTorch implementation of Phillip Isola's `pix2pix` conditional GAN. 

Paper Details: https://arxiv.org/abs/1611.07004

Implementation done for EECS 545, FA21. 

## Steps for training on Great Lakes Cluster:
1. Connect to umich vpn (I usually choose all traffic, idk if it matters)

2. ssh to the cluster by running: `ssh greatlakes.arc-ts.umich.edu`

3. set up your workspace. first add python w/ anaconda by loading a module: `ml load python3.8-anaconda/2021.05`. save this module set with command: `ml save module545`. finally, add pytorch with:  `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --user`.

4. make a new directory if you want. get the code with: `git clone https://github.com/amm449/eecs545project.git` and move into this directory

5. edit the sbat file for your own purposes. really only need to change the email address...

6. run `sbatch training.sbat`. this submits the job. you can check the status of your job with command: `sq`

## if you want to use your own data:
1. transfer your data to the server with scp: `scp -r <your data> greatlakes.arc-ts.umich.edu:</path/to/save>

2. modify config.py to get the data at the right path
