# image-to-image translation
pytorch implementation of phillip isola's `pix2pix` conditional gan. 

paper details: https://arxiv.org/abs/1611.07004

implementation done for eecs 545, fa21. 

## to run:
* requires python3.8 with cuda 11.3
* run `pip3 install -r requirements.txt`
* add your dataset
* update filepaths in `config.py`
* to train, run: `python train.py`
* to test, run: `python test.py`

## steps for training on great lakes cluster:
1. connect to umich vpn
2. ssh to the cluster by running: `ssh greatlakes.arc-ts.umich.edu`
3. set up your workspace. first add python w/ anaconda by loading a module: `ml load python3.8-anaconda/2021.05`. save this module set with command: `ml save module545`. finally, add pytorch with:  `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --user`.
4. make a new directory if you want. clone repository and move into this directory
5. edit the sbat file for your own purposes. really only need to change the email address...
6. run `sbatch training.sbat`. this submits the job. you can check the status of your job with command: `sq`

### if you want to use your own data:
1. transfer your data to the server with scp: `scp -r <your data> greatlakes.arc-ts.umich.edu:</path/to/save>`
2. modify config.py to get the data at the right path

### training with jupyter lab:
1. connect to umich vpn
2. go to https://greatlakes.arc-ts.umich.edu/pun/sys/dashboard/batch_connect/sys/arcts_jupyter_lab/session_contexts/new
3. select the module "python3.8-anaconda/2021.05"
4. enter our slurm account: "eecs545f21_class"
5. for partition, either choose "gpu" or "spgpu"
6. enter your choice of hours
7. you need 1 core, 16gb of memory, and 1 gpu
8. the rest you can leave blank. click launch 
9. using terminal feature you can move around and also git clone our directory

