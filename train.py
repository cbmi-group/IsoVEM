# ----------
#  Configs
# ----------
import sys
import argparse
from utils import add_dict_to_argparser
from dataset import *
from model import *
from metric import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import traceback

def train_func(args, stdout=None):
    try:
        if stdout is not None:
            save_stdout = sys.stdout
            save_stderr = sys.stderr
            sys.stdout = stdout
            sys.stderr = stdout

        torch.cuda.set_device(int(args.train_gpu_ids))
        args.tblogger=os.path.join(args.train_output_dir, "tblogger")
        args.checkpoint=os.path.join(args.train_output_dir, "checkpoint")
        args.visual=os.path.join(args.train_output_dir, "visual")
        os.makedirs(args.tblogger, exist_ok=True)
        os.makedirs(args.checkpoint, exist_ok=True)
        os.makedirs(args.visual, exist_ok=True)

        writer = SummaryWriter(args.tblogger)
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

        # ----------
        #  Generate hyperparams
        # ----------
        print("===> Preparing hyperparameters")
        if args.train_upscale==8:
            # subvolume size
            args.subvol_shape = (16, 128, 128)
            # transformer params
            args.img_size = [16, 16, 128]  # video sequence size
            args.window_size = [2, 2, 16]  # window attention
            args.depths = [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2]  # number of TMSA in TRSA
            args.indep_reconsts = [11, 12]  # per-frame window attention in the end
            args.embed_dims = [40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60, 60, 60]  # feature channels
            args.num_heads = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # number of heads for window attention
        elif args.train_upscale == 10:
            # subvolume size
            args.subvol_shape = (16, 160, 160)
            # transformer params
            args.img_size = [16, 16, 160],  # video sequence size
            args.window_size = [4, 2, 20],  # window attention
            args.depths = [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2],  # number of TMSA in TRSA
            args.indep_reconsts = [11, 12],  # per-frame window attention in the end
            args.embed_dims = [40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60, 60, 60],  # feature channels
            args.num_heads = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # number of heads for window attention
        else:
            raise ValueError(f'Not support the upscale factor {args.train_upscale}')

        # ----------
        #  Datasets
        # ----------
        print("===> Loading datasets")
        # loading training dataset
        train_dataloader = DataLoader(
            ImageDataset_train(image_pth=args.train_data_pth, image_split=args.train_data_split, subvol_shape=args.subvol_shape, scale_factor=args.train_upscale, is_inpaint=args.train_inpaint),
            batch_size=args.train_bs,
            shuffle=True,
        )
        # loading validation dataset
        val_dataloader = DataLoader(
            ImageDataset_val(image_pth=args.train_data_pth, image_split=args.train_data_split, subvol_shape=args.subvol_shape, scale_factor=args.train_upscale, is_inpaint=args.train_inpaint),
            batch_size=args.train_bs,
            shuffle=False,
        )

        # ----------
        #  Model
        # ----------
        print("===> Building models")
        if args.train_is_resume: # loading pretrained model
            model=torch.load(args.train_resume_ckpt_path)
            args.start_epoch = int(args.pretrain_model.split('/')[-1].split('.')[-2].split('_')[-1]) + 1
        else: # create a new model
            model = IsoVEM(upscale=args.train_upscale,
                           img_size=args.img_size,
                           window_size=args.window_size,
                           depths=args.depths,
                           indep_reconsts=args.indep_reconsts,
                           embed_dims=args.embed_dims,
                           num_heads=args.num_heads)
            args.start_epoch = 1

        # create optimizer
        print("===> Setting Optimizer")
        optimizer_G = optim.Adam(model.parameters(), lr=args.train_lr, betas=(0.9, 0.999))

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
        loss_record = np.zeros(shape=(args.train_epoch, 6)) # recording losses and metrics
        for epoch in range(args.start_epoch, args.train_epoch + 1):
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
                gen_hr = model(imgs_lr, current_scale=args.train_upscale)
                # gen_hr = 0.5 * (gen_hr + 1.0)

                # --------------
                #  Loss function
                # --------------
                loss_l1 = torch.nn.L1Loss()(gen_hr, imgs_hr)
                loss_ssim = 1 - ssim(gen_hr, imgs_hr, data_range=1)
                loss_G = loss_l1 + loss_ssim
                loss_G.backward()
                optimizer_G.step()

                # --------------
                #  Log Progress for Model Training
                # --------------
                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d] [ssim loss: %f] [l2 loss: %f] \n"
                    % (epoch, args.train_epoch, iter, len(train_dataloader), loss_ssim.item(), loss_l1.item())
                )

                loss_total_ls[0] += loss_G.item()
                loss_total_ls[1] += loss_l1.item()
                loss_total_ls[2] += loss_ssim.item()
                # break

            loss_total_ls=[x/len(train_dataloader) for x in loss_total_ls]
            loss_record[epoch-1,0:3] = loss_total_ls[:]

            # --------------
            #  Save checkpoints.
            # --------------
            if args.train_ckpt_interval != -1 and epoch % args.train_ckpt_interval == 0:
                torch.save(model, os.path.join(args.checkpoint, "model_epoch_{}.pth".format(epoch)))
                print("Checkpoint of epoch {} is saved to {}".format(epoch, args.checkpoint))

            # ----------
            #  Validation
            # ----------
            with torch.no_grad():
                model.eval()
                val_metirc_ls = [ 0, 0]
                for val_iter, val_batch in enumerate(val_dataloader):
                    # --------------
                    #  Read data
                    # --------------
                    val_img_hr = Variable(val_batch["hr"].type(Tensor))
                    val_img_lr = Variable(val_batch["lr"].type(Tensor))

                    # --------------
                    #  Perform x/y-axis isotropic reconstruction
                    # --------------
                    # val_img_lr = transforms.Normalize([0.5], [0.5])(val_img_lr)
                    val_gen_hr = model(x=val_img_lr, current_scale=args.train_upscale)
                    # val_gen_hr =0.5 * (val_gen_hr + 1.0)

                    val_ssim = compute_ssim(val_gen_hr, val_img_hr, need_2d=False)[0]
                    val_psnr = compute_psnr(val_gen_hr, val_img_hr, need_2d=False)[0]

                    # --------------
                    #  Save visualization results.
                    # --------------
                    if val_iter==0:
                        image_savedir =os.path.join(args.visual, "%04d" % epoch + '.tif')
                        val_gen_hr_np=val_gen_hr[0].squeeze().float().cpu().clamp_(0, 1).numpy()
                        io.imsave(image_savedir,(val_gen_hr_np*255).astype('uint8'))
                        print('Validation volume of epoch {} is saved to {}'.format(epoch, image_savedir))

                    # --------------
                    #  Save validation metrics.
                    # --------------
                    val_metirc_ls[0] += val_ssim
                    val_metirc_ls[1] += val_psnr
                    # break

            val_metirc_ls = [x / len(val_dataloader) for x in val_metirc_ls]
            loss_record[epoch-1, 4:6] = val_metirc_ls[:]

            # --------------
            #  Log Progress for Model Validation
            # --------------
            sys.stdout.write(
                "[Valid_Epoch %d/%d] [valid_ssim: %f] [valid_psnr: %f] \n"
                % (epoch, args.train_epoch, val_metirc_ls[0], val_metirc_ls[1])
            )

            # ----------
            #  Log to Tensorboard
            # ----------
            np.savetxt(os.path.join(args.train_output_dir, "loss_metric.csv"), loss_record, delimiter=',')
            writer.add_scalar('loss_Generator', loss_total_ls[0], epoch)
            writer.add_scalar('loss_L1', loss_total_ls[1], epoch)
            writer.add_scalar('loss_SSIM', loss_total_ls[2], epoch)
            writer.add_scalar('val_ssim', val_metirc_ls[0], epoch)
            writer.add_scalar('val_psnr', val_metirc_ls[1], epoch)
        print('*' * 100)
        if stdout is not None:
            sys.stdout = save_stdout
            sys.stderr = save_stderr
    except:
        print('*' * 100)
        if stdout is not None:
            sys.stdout = save_stdout
            sys.stderr = save_stderr
        traceback.print_exc()
        # print(f"{ex}")
        print('*' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for IsoVEM Training')
    parser.add_argument('--train_config_path', help='path of train config file', type=str,
                        default="configs/demo_train.json")

    with open(parser.parse_args().train_config_path, 'r', encoding='UTF-8') as f:
        train_config = json.load(f)
    add_dict_to_argparser(parser, train_config)
    args = parser.parse_args()
    print(args)

    train_func(args, None)
