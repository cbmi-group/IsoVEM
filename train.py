# ----------
#  Configs
# ----------
print("===> Loading configurations")
import sys
import configs
opt = configs.get_EPFL_configs()
print(opt)

# ----------
#  Preparing
# ----------
print("===> Preparing environment")
from dataset import *
from model import *
from metric import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter

writer = SummaryWriter(opt.tblogger)
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
os.makedirs(os.path.join(opt.ckpt_dir,opt.exp_name),exist_ok=True)
os.makedirs(os.path.join(opt.valid_dir,opt.exp_name),exist_ok=True)

# ----------
#  Datasets
# ----------
print("===> Loading datasets")
# loading training dataset
train_dataloader = DataLoader(
    ImageDataset_train(image_h5=opt.train_h5, image_shape=opt.train_shape, scale_factor=opt.upscale, is_gt=opt.train_gt, is_inpaint=opt.inpaint),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# loading validation dataset
val_dataloader = DataLoader(
    ImageDataset_val(image_h5=opt.val_h5, image_shape=opt.val_shape, scale_factor=opt.upscale, is_gt=opt.val_gt, is_inpaint=opt.inpaint),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# ----------
#  Model
# ----------
print("===> Building models")
if opt.pretrain_model: # loading pretrained model
    model=torch.load(opt.pretrain_model)
    opt.start_epoch = int(opt.pretrain_model.split('/')[-1].split('.')[-2].split('_')[-1])+ 1
else: # create a new model
    model = IsoVEM(upscale=opt.upscale,
                img_size=opt.img_size,
                window_size=opt.window_size,
                depths=opt.depths,
                indep_reconsts=opt.indep_reconsts,
                embed_dims=opt.embed_dims,
                num_heads=opt.num_heads)

# create optimizer
print("===> Setting Optimizer")
optimizer_G = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# load to cuda
if cuda:
    print("===> Setting CUDA")
    model = model.cuda()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


# ----------
#  Training
# ----------
print("===> Training")
loss_record = np.zeros(shape=(opt.end_epoch, 8)) # recording losses and metrics
for epoch in range(opt.start_epoch, opt.end_epoch+1):
    model.train()

    loss_total_ls=[0,0,0]
    print("Epoch={}, lr={}".format(epoch, optimizer_G.param_groups[0]["lr"]))

    for iter, batch in enumerate(train_dataloader, 1):
        # --------------
        #  Read data
        # --------------
        imgs_hr = Variable(batch["hr"].type(Tensor))
        imgs_lr = Variable(batch["lr"].type(Tensor))

        optimizer_G.zero_grad()

        # --------------
        #  Perform x/y-axis isotropic reconstruction
        # --------------
        # imgs_lr=transforms.Normalize([0.5], [0.5])(imgs_lr)
        gen_hr = model(imgs_lr,current_scale=opt.upscale)
        # gen_hr = 0.5 * (gen_hr + 1.0)

        # --------------
        #  Loss function
        # --------------
        loss_l2 = torch.nn.L1Loss()(gen_hr, imgs_hr)
        loss_ssim = 1 - ssim(gen_hr, imgs_hr, data_range=1)
        loss_G = loss_l2 + loss_ssim
        loss_G.backward()
        optimizer_G.step()

        # --------------
        #  Log Progress for Model Training
        # --------------
        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [ssim loss: %f] [l2 loss: %f] \n"
            % (epoch, opt.end_epoch, iter, len(train_dataloader),loss_ssim.item(), loss_l2.item())
        )

        loss_total_ls[0] += loss_G.item()
        loss_total_ls[1] += loss_ssim.item()
        loss_total_ls[2] += loss_l2.item()
        # break

    loss_total_ls=[x/len(train_dataloader) for x in loss_total_ls]
    loss_record[epoch-1,0:3] = loss_total_ls[:]

    # --------------
    #  Save checkpoints.
    # --------------
    if opt.ckpt_interval != -1 and epoch % opt.ckpt_interval == 0:
        model_save_path=os.path.join(opt.ckpt_dir,opt.exp_name)
        torch.save(model, os.path.join(model_save_path,"model_epoch_{}.pth".format(epoch)))
        print("Checkpoint of epoch {} is saved to {}".format(epoch,model_save_path))

    # ----------
    #  Validation
    # ----------
    with torch.no_grad():
        model.eval()
        val_metirc_ls = [0, 0, 0, 0]
        for val_iter, val_batch in enumerate(val_dataloader):
            # --------------
            #  Read data
            # --------------
            val_img_hr_0 = Variable(val_batch["gt"].type(Tensor))
            val_img_hr_1 = Variable(val_batch["hr"].type(Tensor))
            val_img_hr_2 = Variable(val_batch["lr"].type(Tensor))

            # --------------
            #  Perform x/y-axis isotropic reconstruction
            # --------------
            # val_img_hr_2 = transforms.Normalize([0.5], [0.5])(val_img_hr_2)
            val_gen_hr_2 = model(x=val_img_hr_2, current_scale=opt.upscale)
            # val_gen_hr_2 =0.5 * (val_gen_hr_2 + 1.0)

            val_ssim_2 = compute_ssim(val_gen_hr_2, val_img_hr_1, need_2d=False)[0]
            val_psnr_2 = compute_psnr(val_gen_hr_2, val_img_hr_1, need_2d=False)[0]

            # --------------
            #  When gt exists, perform z-axis isotropic reconstruction
            # --------------
            if opt.val_gt:
                val_gen_hr_1 = model(x=val_img_hr_1.permute(0,3,2,1,4),current_scale=opt.upscale)
                val_ssim_1 = compute_ssim(val_gen_hr_1, val_img_hr_0.permute(0,3,2,1,4), need_2d=False)[0]
                val_psnr_1 = compute_psnr(val_gen_hr_1, val_img_hr_0.permute(0,3,2,1,4), need_2d=False)[0]
            else:
                val_gen_hr_1 = val_gen_hr_2
                val_ssim_1 = torch.tensor(0)
                val_psnr_1 = torch.tensor(0)

            # --------------
            #  Save visualization results.
            # --------------
            if val_iter==0:
                image_savedir =os.path.join(opt.valid_dir, opt.exp_name, "%04d" % epoch+'.tif')
                val_gen_hr_1_np=val_gen_hr_1.permute(0,3,2,1,4)[0,:,:,:,:].squeeze().float().cpu().clamp_(0, 1).numpy()
                io.imsave(image_savedir,(val_gen_hr_1_np*255).astype('uint8'))
                print('Validation volume of epoch {} is saved to {}'.format(epoch, image_savedir))

            # --------------
            #  Save validation metrics.
            # --------------
            val_metirc_ls[0] += val_ssim_1
            val_metirc_ls[1] += val_psnr_1
            val_metirc_ls[2] += val_ssim_2
            val_metirc_ls[3] += val_psnr_2
            # break

    val_metirc_ls = [x / len(val_dataloader) for x in val_metirc_ls]
    loss_record[epoch-1, 4:8] = val_metirc_ls[:]

    # --------------
    #  Log Progress for Model Validation
    # --------------
    sys.stdout.write(
        "[Valid_Epoch %d/%d] [val1_ssim: %f] [val1_psnr: %f] [val2_ssim: %f] [val2_psnr: %f]"
        % (epoch, opt.end_epoch, val_metirc_ls[0], val_metirc_ls[1],val_metirc_ls[2], val_metirc_ls[3])
    )

    # ----------
    #  Log to Tensorboard
    # ----------
    np.savetxt(os.path.join(opt.ckpt_dir,opt.exp_name,"loss_metric.csv"), loss_record, delimiter=',')
    writer.add_scalar('loss_Generator', loss_total_ls[0], epoch)
    writer.add_scalar('loss_SSIM', loss_total_ls[1], epoch)
    writer.add_scalar('loss_L1', loss_total_ls[2], epoch)
    writer.add_scalar('val1_ssim', val_metirc_ls[0], epoch)
    writer.add_scalar('val1_psnr', val_metirc_ls[1], epoch)
    writer.add_scalar('val2_ssim', val_metirc_ls[2], epoch)
    writer.add_scalar('val2_psnr', val_metirc_ls[3], epoch)
