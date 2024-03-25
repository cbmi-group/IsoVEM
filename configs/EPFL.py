import ml_collections
import os


def get_EPFL_configs():
  config = ml_collections.ConfigDict()

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

  return config