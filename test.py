import argparse
from dataset import *
from metric import *
from utils import *
from tqdm import tqdm
import sys

def run_model_inpaint(model,image,idx_ls, args):
    '''
    model inference for slice inpainting.
    :param model: pretrained isovem model
    :param image: test data wait for processing
    :param idx_ls: the list of slice index (start from 0) along Z-axis to be inpainted.
    '''
    # ----------
    #  Crop Subvolumes
    # ----------
    coord_np, y_crop_num, x_crop_num = create_coord_2d((image.shape[1],image.shape[2]), (args.test_shape[1], args.test_shape[2]), args.test_overlap)

    pred_inpaint_ls = []
    # for each slice
    for idx in idx_ls:
        vol_ls = []
        # ----------
        #  Model Inference
        # ----------
        with torch.no_grad():
            # for each region
            for i in tqdm(range(coord_np.shape[1]), desc='Inpaint slice '+str(idx)):
                y, x = coord_np[0, i], coord_np[1, i]
                crop = np.s_[idx - args.test_shape[0] // 2:idx + args.test_shape[0] // 2, y - args.test_shape[1] // 2:y + args.test_shape[1] // 2,
                       x - args.test_shape[2] // 2:x + args.test_shape[2] // 2]
                batch = image[crop]
                # model inference
                batch = torch.tensor(batch / 255.0).type(Tensor).unsqueeze(1).unsqueeze(0)
                pred = model(x=batch,current_scale=1)
                pred=pred.squeeze().cpu().detach().numpy()
                # save inpainted region
                vol_ls.append(pred[args.test_shape[0] // 2])
        # ----------
        #  2D Stitch
        # ----------
        print("Stitching takes some time...")
        stitch_vol=stitch2D(vol_ls, (image.shape[1],image.shape[2]), (args.test_shape[1], args.test_shape[2]), args.test_overlap)
        pred_inpaint_ls.append(float2uint8(stitch_vol))

    return pred_inpaint_ls


def run_model_isosr(model,image,scale_factor):
    '''
    model inference for isotropic reconstruction with uncertainty map.
    :param model: pretrained isovem model
    :param image: test data wait for processing
    :param scale_factor: the scale factor of isotropic reconstruction.
    '''
    # ----------
    #  Crop Subvolumes
    # ----------
    coord_np,z_crop_num,y_crop_num,x_crop_num = create_coord_3d(image.shape, args.test_shape, args.test_overlap)

    vol_pred_ls=[]
    vol_uncertainty_ls = []
    # ----------
    #  Model Inference
    # ----------
    with torch.no_grad():
        # for each subvolume
        for i in tqdm(range(coord_np.shape[1]), desc='IsoSR'):
            z, y, x = coord_np[0, i], coord_np[1, i], coord_np[2, i]
            crop = np.s_[z - args.test_shape[0] // 2:z + args.test_shape[0] // 2, y - args.test_shape[1] // 2:y + args.test_shape[1] // 2,
                   x - args.test_shape[2] // 2:x + args.test_shape[2] // 2]
            batch = image[crop]
            # eight rotations
            batch = torch.tensor(batch[np.newaxis, np.newaxis, ...] / 255.0).type(Tensor)
            batch_rot_ls = rotate_8(batch)
            pred_rot_ls = []
            # for each rotated subvolume
            for j in range(0, len(batch_rot_ls)):
                batch_rot = batch_rot_ls[j]
                # model inference
                batch_rot=batch_rot.squeeze().permute(1, 0, 2).unsqueeze(1).unsqueeze(0)
                pred_rot = model(x=batch_rot,current_scale=scale_factor)
                pred_rot =pred_rot.squeeze().permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
                pred_rot_ls.append(pred_rot)
            # save isosr
            pred_ls, pred = inv_rotate_8(pred_rot_ls, (pred_rot.shape[2], pred_rot.shape[3], pred_rot.shape[4]))
            pred_ls = [item.squeeze().cpu().detach().numpy() for item in pred_ls]
            uncertainty = np.std(np.array(pred_ls), axis=0)
            pred = pred.squeeze().cpu().detach().numpy()

            vol_pred_ls.append(pred)
            vol_uncertainty_ls.append(uncertainty)

    # ----------
    #  Free Memory
    # ----------
    del pred_rot
    del batch_rot
    del pred_rot_ls
    del batch_rot_ls
    del pred
    del batch
    torch.cuda.empty_cache()

    # ----------
    #  3D stitch
    # ----------
    print("Stitching takes some time...")
    stitch_isosr = stitch3D(vol_pred_ls, image.shape, args.test_shape, args.test_overlap, args.test_upscale)
    stitch_isosr_map = stitch3D(vol_uncertainty_ls, image.shape, args.test_shape, args.test_overlap, args.test_upscale)
    return float2uint8(stitch_isosr),float2uint8(stitch_isosr_map)

def test_func(args, stdout=None):
    if stdout is not None:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stdout

    print("===> Preparing environment")
    torch.cuda.set_device(int(args.test_gpu_ids))
    os.makedirs(args.test_output_dir, exist_ok=True)
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # ----------
    #  Load data and model
    # ----------
    if args.test_data_pth.split('.')[-1] == 'tif':
        image_file = io.imread(args.test_data_pth)
    elif args.test_data_pth.split('.')[-1] == 'h5':
        image_file = np.array(h5py.File(args.test_data_pth, 'r')['raw'])
    else:
        raise ValueError(f'Not support the image format of {args.test_data_pth}')

    device=torch.device("cuda")
    model = torch.load(args.test_ckpt_path,map_location=device).eval()

    # ----------
    #  Generate hyperparams
    # ----------
    if args.test_upscale == 8:
        args.test_shape = (16, 128, 128)
        args.test_overlap = 8
    elif args.test_upscale == 10:
        args.test_shape = (16, 160, 160)
        args.test_overlap = 8
    else:
        raise ValueError(f'Not support the upscale factor {args.train_upscale}')

    # ----------
    #  Model Inference for slice inpainting
    # ----------
    if args.test_inpaint:
        pred_inpaint_ls = run_model_inpaint(model, image_file, args.test_inpaint_index, args)
        for i, idx in enumerate(args.test_inpaint_index):
            image_file[idx] = pred_inpaint_ls[i]
        io.imsave(os.path.join(args.test_output_dir, 'inpaint.tif'), image_file)
        print('Inpainted input volume is saved to {}, named {}'.format(args.test_output_dir, 'inpaint.tif'))

    # ----------
    #  Model Inference for isotropic reconstruction with uncertainty map
    # ----------
    pred_isosr, pred_isosr_map = run_model_isosr(model, image_file, args.test_upscale)
    io.imsave(os.path.join(args.test_output_dir, 'isosr.tif'), pred_isosr)
    print('Isotropic reconstructed volume is saved to {}, named {}'.format(args.test_output_dir, 'isosr.tif'))
    io.imsave(os.path.join(args.test_output_dir, 'isosr_map.tif'), pred_isosr_map)
    print('Uncertainty map of reconstruction is saved to {}, named {}'.format(args.test_output_dir, 'isosr_map.tif'))

    print('*' * 100)
    if stdout is not None:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for IsoVEM Testing')
    parser.add_argument('--test_config_path', help='path of test config file', type=str,
                        default="configs/demo_test.json")

    with open(parser.parse_args().test_config_path, 'r', encoding='UTF-8') as f:
        test_config = json.load(f)
    add_dict_to_argparser(parser, test_config)
    args = parser.parse_args()
    print(args)

    test_func(args)

