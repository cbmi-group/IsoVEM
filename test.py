import itertools
from tqdm import tqdm
import time

def get_crop_num(length,crop_size=128,overlap=16):
    '''
    calculate the number of overlapping cropping
    length=n*crop_size-(n-1)*overlap
    '''
    num=(length-overlap)/(crop_size-overlap)
    return math.ceil(num)

def create_coord(s1,s2=(128,128,128),overlap=16):
    '''
    calculate the cropping coordinates to extract subvolumes
    '''
    coord_ls=[[],[],[]]
    z_crop_num=get_crop_num(s1[0],crop_size=s2[0],overlap=overlap)
    y_crop_num=get_crop_num(s1[1],crop_size=s2[1],overlap=overlap)
    x_crop_num=get_crop_num(s1[2],crop_size=s2[2],overlap=overlap)

    for z,y,x in itertools.product(range(z_crop_num), range(y_crop_num),range(x_crop_num)):
        z_coord = s2[0]//2 + z * (s2[0] - overlap) if (s2[0]//2 + z * (s2[0] - overlap)) < s1[0] - s2[0]//2 else s1[0] - s2[0]//2
        y_coord = s2[1]//2 + y * (s2[1] - overlap) if (s2[1]//2 + y * (s2[1] - overlap)) < s1[1] - s2[1]//2 else s1[1] - s2[1]//2
        x_coord = s2[2]//2 + x * (s2[2] - overlap) if (s2[2]//2 + x * (s2[2] - overlap)) < s1[2] - s2[2]//2 else s1[2] - s2[2]//2
        coord_ls[0].append(z_coord)
        coord_ls[1].append(y_coord)
        coord_ls[2].append(x_coord)

    return np.array(coord_ls),z_crop_num,y_crop_num,x_crop_num


def run_model(model,test_h5,scale_factor,image_shape=(128,128,128),overlap=16,tta=True,uncertainty=False,debris=False):
    '''
    model inference.
    :param model: pretrained isovem/isovem+ model
    :param test_h5: test data wait for processing
    :param scale_factor: the scale factor of isotropic reconstruction, it can be different from training phase.
    :param image_shape: the subvolume size for model inference.
    :param overlap: the overlapped voxel for subvolumes stitching
    :param tta: if true, perform test time augmentation based on 8 orthogonal rotations
    :param uncertainty: if true, get model uncertainty map rather than isotropic reconstruction results. need to set test_tta=True.
    :param debris: if true, perform slice inpainting, need to set test_tta=True and test_upscale=1. if false, perform isotropic reconstruction.

    case1: tta=True,uncertainty=False,debris=False, perform isotropic reconstruction, for IsoVEM or IsoVEM+'s stage2.
    case2: tta=True,uncertainty=True,debris=False, generate model uncertainty map for fidelity evaluation.
    case3: tta=True,uncertainty=False,debris=True, perform slice inpainting for IsoVEM+ stage1.
    case4: tta=False,uncertainty=False,debris=False, perform simpler isotropic reconstruction.
    '''

    # ----------
    #  Crop Subvolumes
    # ----------
    coord_np,z_crop_num,y_crop_num,x_crop_num = create_coord(test_h5.shape, image_shape,overlap)
    test_shape = test_h5.shape
    vol_ls=[]
    with torch.no_grad():
        for i in tqdm(range(coord_np.shape[1]), desc='Running'):
            z, y, x = coord_np[0, i], coord_np[1, i], coord_np[2, i]
            crop = np.s_[z - image_shape[0]//2:z + image_shape[0]//2, y - image_shape[1]//2:y + image_shape[1]//2,
                   x - image_shape[2]//2:x + image_shape[2]//2]
            batch = test_h5[crop]
            batch = torch.tensor(batch[np.newaxis, np.newaxis, ...] / 255.0).type(Tensor)

            # ----------
            #  Inference on Subvolumes
            # ----------
            if tta:# test-time augmentation by 8 rotations
                batch_rot_ls = rotate_8(batch)
                pred_rot_ls = []

                for j in range(0, len(batch_rot_ls)): # for each rotated subvolume
                    batch_rot = batch_rot_ls[j]
                    if debris: # if true, change processing axis
                        batch_rot = batch_rot.permute(0, 1, 3, 2, 4)

                    # model inference
                    batch_rot=batch_rot.squeeze().permute(1, 0, 2).unsqueeze(1).unsqueeze(0)
                    assert(scale_factor==1 if debris else True)
                    pred_rot = model(x=batch_rot,current_scale=scale_factor)
                    pred_rot =pred_rot.squeeze().permute(1,0,2).unsqueeze(0).unsqueeze(0)

                    if debris: # if true, change processing axis
                        pred_rot = pred_rot.permute(0, 1, 3, 2, 4)
                    pred_rot_ls.append(pred_rot)
                image_shape_=(pred_rot.shape[2],pred_rot.shape[3],pred_rot.shape[4])

                if uncertainty:# visualize model uncertainty map
                    pred_ls,_=anti_rotate_8(pred_rot_ls, image_shape_)
                    pred_ls=[item.squeeze().cpu().detach().numpy() for item in pred_ls]
                    pred=np.std(np.array(pred_ls),axis=0)
                else:# visualize isotropic reconstruction results
                    _, pred = anti_rotate_8(pred_rot_ls, image_shape_)
                    pred = pred.squeeze().cpu().detach().numpy()

            else:# no TTA
                # model inference
                batch = batch.squeeze().permute(1, 0, 2).unsqueeze(1).unsqueeze(0)
                pred = model(x=batch,current_scale=scale_factor)
                pred=pred.squeeze().permute(1,0,2).cpu().detach().numpy()
            vol_ls.append(pred)

        # ----------
        #  Free Memory
        # ----------
        if tta:
            del pred_rot
            del batch_rot
            del pred_rot_ls
            del batch_rot_ls
        del pred
        del batch
        del model
        del test_h5
        torch.cuda.empty_cache()

    # ----------
    #  Save Intermediate Results
    # ----------
    # vol_np=np.array(vol_ls)
    # np.save(os.path.join(save_dir,'pred.npy'),vol_np)
    # del vol_np
    # vol_np=np.load(os.path.join(save_dir,'pred.npy'))
    # vol_ls=vol_np.tolist()

    # ----------
    #  3D Stitch
    # ----------
    # 3d stitch along x axis
    x_ls=[]
    for i in range(z_crop_num):
        for j in range(y_crop_num):
            x_temp = None
            for k in range(x_crop_num):
                nps=np.s_[i * y_crop_num * x_crop_num + j * x_crop_num + k]
                if x_temp is not None:
                    ovlp=overlap if x_temp.shape[2]+vol_ls[nps].shape[2]-overlap<=test_shape[2] \
                        else x_temp.shape[2]+vol_ls[nps].shape[2]-test_shape[2]
                    x_temp = blend3D_X(x_temp, vol_ls[nps],overlap=ovlp)
                else:
                    x_temp = vol_ls[nps]
            x_ls.append(x_temp)
    # 3d stitch along y axis
    y_ls=[]
    for i in range(z_crop_num):
        y_temp = None
        for j in range(y_crop_num):
            nps =np.s_[i * y_crop_num + j]
            if y_temp is not None:
                ovlp = overlap if y_temp.shape[1] + x_ls[nps].shape[1] - overlap <= test_shape[1] \
                    else y_temp.shape[1] + x_ls[nps].shape[1] - test_shape[1]
                y_temp = blend3D_Y(y_temp, x_ls[nps],overlap=ovlp)
            else:
                y_temp = x_ls[nps]
        y_ls.append(y_temp)
    # 3d stitch along z axis
    z_temp=None
    for i in range(z_crop_num):
        if z_temp is not None:
            ovlp = int(overlap* scale_factor) if z_temp.shape[0] + y_ls[i].shape[0] - int(overlap* scale_factor) <= int(test_shape[0] * scale_factor) \
                else z_temp.shape[0] + y_ls[i].shape[0] - int(test_shape[0] * scale_factor)
            z_temp = blend3D_Z(z_temp, y_ls[i],overlap=ovlp)
        else:
            z_temp = y_ls[i]

    # ----------
    #  Return Final Results
    # ----------
    z_temp[z_temp < 0] = 0
    z_temp[z_temp > 1] = 1
    res=(z_temp* 255).astype('uint8')
    return res


if __name__ == '__main__':
    # ----------
    #  Configs
    # ----------
    import configs
    opt = configs.get_EPFL_configs()
    print(opt)

    # ----------
    #  Preparing
    # ----------
    from dataset import *
    from metric import *
    from utils import *

    # device
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # save path
    save_dir = os.path.join(opt.test_dir,opt.test_model.split('/')[-2]+'_'+opt.test_model.split('/')[-1].split('.')[0])
    os.makedirs(save_dir, exist_ok=True)
    pred_name=os.path.join(save_dir,'pred.tif')
    print(pred_name)

    # ----------
    #  Load data
    # ----------
    tic = time.time()
    input_h5 = io.imread(opt.test_h5)

    # ----------
    #  Model Inference
    # ----------
    model = torch.load(opt.test_model).eval()
    pred_h5 = run_model(model, input_h5, opt.test_upscale, opt.test_shape, opt.test_overlap,
                        tta=opt.test_tta, uncertainty=opt.test_uncertainty,debris=opt.test_debris)

    # ----------
    #  Save results
    # ----------
    io.imsave(pred_name,pred_h5)
    tac = time.time()
    print("Testing costs{:.2f} min".format((tac - tic) / 60))

    # ----------
    #  If gt exists, perform metric evaluation.
    # ----------
    if opt.test_gt:
        import time
        print('Start:', time.strftime('%Y-%m-%d %H:%M:%S'))
        tic = time.time()

        test_h5 = io.imread(opt.test_gt)
        pred_h5 = io.imread(pred_name)
        assert (pred_h5.shape == test_h5.shape)
        calculate_metrics(pred_h5, test_h5, save_json=pred_name+"_metric.json", is_cuda=False)

        tac = time.time()
        print('End:', time.strftime('%Y-%m-%d %H:%M:%S'))
        print("Testing costs{:.2f} min".format((tac - tic) / 60))