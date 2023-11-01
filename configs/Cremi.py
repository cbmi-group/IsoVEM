import ml_collections
import os


def get_Cremi_configs():
  config = ml_collections.ConfigDict()

  # experiment
  config.exp_name="cremi-10"
  config.upscale=10
  config.inpaint=False

  # running
  config.use_tb_logger=True
  config.tblogger="../TBlogger/"+config.exp_name
  config.gpu_ids="4"
  os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids
  config.n_cpu=0

  # datasets
  config.train_h5 = "/mnt/data1/EMData/Data/Cremi/train_10.h5"
  config.val_h5 = "/mnt/data1/EMData/Data/Cremi/val_10.h5"
  config.train_shape = (16,160,160)
  config.val_shape = (16,160,160)
  config.train_gt = False
  config.val_gt = False

  # paths
  config.ckpt_dir= "/mnt/data1/EMData/D_Ours/6_VRT/ckpt"
  config.valid_dir="/mnt/data1/EMData/D_Ours/6_VRT/valid"
  config.pretrain_model=None

  # model
  config.img_size=[16,16,160] # video sequence size
  config.window_size=[2,2,20] # window attention
  config.depths=[4,4,4,4,4,4,4, 2,2,2,2, 2,2] # number of TMSA in TRSA
  config.indep_reconsts=[11,12] # per-frame window attention in the end
  config.embed_dims=[40,40,40,40,40,40,40, 60,60,60,60, 60,60] # feature channels
  config.num_heads=[4,4,4,4,4,4,4, 4,4,4,4, 4,4] # window attention

  # train
  config.start_epoch=1
  config.end_epoch=300
  config.ckpt_interval=1
  config.batch_size=2
  config.lr=1e-3
  config.b1=0.9
  config.b2=0.999

  # test
  config.test_h5 = "/mnt/data1/EMData/Data/Cremi/test_10.h5"
  config.test_model = "/mnt/data1/EMData/D_Ours/6_VRT/ckpt/cremi-10/model_epoch_1.pth"
  config.test_dir = "/mnt/data1/EMData/D_Ours/6_VRT/test"
  config.test_shape = (16, 160, 160)
  config.test_upscale = 10
  config.test_overlap = 8
  config.test_tta = False
  config.test_artifact = False
  config.test_debris=False
  config.test_gt=None

  return config