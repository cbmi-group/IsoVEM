# IsoVEM: Video Transformer Enables Accurate and Robust Isotropic Reconstruction for Volume Electron Microscopy

This repository `IsoVEM` is the official implementation of the bioRxiv paper(https://www.biorxiv.org/content/10.1101/2023.11.22.567807v3).

## Usage

#### 1.Install dependencies

Here's a summary of the key dependencies.

- python 3.7
- pytorch 1.8.1
- CUDA 11.1

We recommend the following demand to install all of the dependencies.

```
conda create -n isovem
conda activate isovem
pip install -r requirements.txt
```

The typical install time for these packages is within 1 hour.

#### 2.Configuration

The predefined config files are provided in`configs/epfl.py` or  `configs/cremi.py`. The meaning of each argument has been annotated as follows. You can also define a new config file as needed. 

```
# experiment
config.exp_name="epfl-8" # experiment name
config.upscale=8 # scale factor during training
config.inpaint=False # if True, perform IsoVEM+ training

# running
config.gpu_ids="6" # gpu device
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids
config.n_cpu = 0

# tblogger
config.use_tb_logger=True
config.tblogger="tblogger/"+config.exp_name # tensorboard logger path

# datasets
config.train_h5 = "data/epfl/train_gt.tif" # training data path
config.val_h5 = "data/epfl/val_gt.tif" # validation data path
config.train_shape = (128,128,128) # training subvolume size
config.val_shape = (128,128,128) # validation subvolume size
config.train_gt = True # whether train_h5 is an isotropic data
config.val_gt = True # whether val_h5 is an isotropic data

# paths
config.ckpt_dir= "ckpt" # saving path for model checkpoints
config.valid_dir="valid" # saving path for visualizations during validation
config.pretrain_model=None # pretrained model path. if None, training from scratch.

# model
config.img_size=[16,16,128] # video sequence size
config.window_size=[2,2,16] # window attention
config.depths=[4,4,4,4,4,4,4, 2,2,2,2, 2,2] # number of TMSA in TRSA
config.indep_reconsts=[11,12] # per-frame window attention in the end
config.embed_dims=[40,40,40,40,40,40,40, 60,60,60,60, 60,60] # feature channels
config.num_heads=[4,4,4,4,4,4,4, 4,4,4,4, 4,4] # number of heads for window attention

# train
config.start_epoch=1  # starting training epoch
config.end_epoch=200 # ending training epoch
config.ckpt_interval=1 # epoch interval for saving checkpoints
config.batch_size=2 # batch size
config.lr=1e-3 # learning rate
config.b1=0.9  # adam optimizer params
config.b2=0.999 # adam optimizer params

# test
config.test_h5 = "data/epfl/test_8.tif" # test anisotropic data path
config.test_model = "ckpt/epfl-8/model_epoch_1.pth" # model checkpoint for testing
config.test_dir = "test" # saving path for testing results
config.test_shape = (16, 128, 128) # subvolume size
config.test_overlap = 8 # overlapped voxels for subvolume stitching
config.test_upscale = 8 # scale factor during inference, it can be different from training phase
config.test_tta = True # if true, perform test time augmentation based on 8 orthogonal rotations
config.test_uncertainty = False # if true, get model uncertainty map rather than isotropic reconstruction results. need to set test_tta=True.
config.test_debris=False # if true, perform slice inpainting, need to set test_upscale=1. if false, perform isotropic reconstruction
config.test_gt="data/epfl/test_gt.tif" # test isotropic data path for metrics calculation
```

#### 3.Model Training/Testing

After define the configuration file, the training and testing can be performed on  train.py  and test.py, only change configure importation.

```
import configs
opt = configs.get_EPFL_configs()
```

Run the training code as follows. The typical training time for demo data costs several hours. 

```
python train.py 
```

Run the testing code as follows. The typical testing time for demo data costs 20 mins.

```
python test.py 
```

#### 4.Attention Visualization

Visualizing the attention module in IsoVEM helps to better understand model's behavior. The tool [Visualizer](https://github.com/luo3300612/Visualizer) needs to be installed as its instruction to generate the attention map of intermediate layer of isoVEM. 

The interested layer and attention window can be defined customly.

```
roi = np.s_[70:70 + 64, 56:56 + 16, 1138:1138 + 256] # roi of input volume, such as region containing bilayer
idx_depth=3 # idx range: the length of config.depths list. choose self-attention-only and no-downsample layer in TMSAG
window_size = [4,8,32] # window size for visualization
idx_window=167 # idx range: img_size=[16,64,256]//window_size=[4,8,32]=256
idx_query=84 # idx range: window_size=[4,8,32], 8*32=256
```

After installing the [Visualizer](https://github.com/luo3300612/Visualizer) tool, run the visualization code as follows.

```
python models/attmap.py 
```

## Acknowledgement

The network code is based on [VRT(Video Restoration Transformer)](https://arxiv.org/abs/2201.12288) and its official [implementation](https://github.com/JingyunLiang/VRT/tree/main). We thank the authors for their work and for sharing the code.

## Citation

If you find this repository useful in your research, please cite our paper:

```
@article {He2023.11.22.567807,
	author = {Jia He and Yan Zhang and Wenhao Sun and Ge Yang and Fei Sun},
	title = {IsoVEM: Isotropic Reconstruction for Volume Electron Microscopy Based on Transformer},
	elocation-id = {2023.11.22.567807},
	year = {2023},
	doi = {10.1101/2023.11.22.567807},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/12/27/2023.11.22.567807},
	eprint = {https://www.biorxiv.org/content/early/2023/12/27/2023.11.22.567807.full.pdf},
	journal = {bioRxiv}
}
```
