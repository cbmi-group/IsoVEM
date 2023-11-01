# use Visualizer tool to visualize attention map
# https://github.com/luo3300612/Visualizer

import os
import sys
sys.path.append("..")
from dataset import *
from metric import *
import cv2
from visualizer import get_local

def idx2coord(idx,window_size):
    a,b,c=window_size
    coord_a=idx//(b*c)
    coord_b=idx%(b*c)//c
    coord_c=idx%(b*c)%c
    return (coord_a,coord_b,coord_c)

def norm(x):
    x=(x-x.min())/(x.max()-x.min()+1e-6)
    return x

def vis_attmap(model,input,save_dir,idx_depth,window_size,idx_window,idx_query):
    # model inference
    with torch.no_grad():
        input_t = torch.tensor(input / 255.0).type(Tensor)
        input_t = input_t.permute(1, 0, 2).unsqueeze(1).unsqueeze(0)
        pred_t = model(input_t)

    # save pred
    pred = pred_t.squeeze().permute(1, 0, 2).detach().cpu().numpy()
    pred=(pred*255).astype('uint8')
    io.imsave(os.path.join(save_dir, 'pred.tif'), pred)

    # save patch
    D, H, W = input_t.squeeze().shape
    input_tw =input_t.squeeze().view(D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],window_size[2])
    input_tw = input_tw.permute(0,2,4,1,3,5)

    # select key structure
    # for idx_window_ in range(0,input_tw.shape[0]*input_tw.shape[1]*input_tw.shape[2]):
    #     coord_window = idx2coord(idx_window_, input_tw.shape[0:3])
    #     input_tw_ = input_tw[coord_window]
    #     image = (input_tw_.detach().cpu().numpy() * 255).astype('uint8')
    #     io.imsave(os.path.join(save_dir, 'img_%04d' % (idx_window_) + '_00.tif'), image[0])
    #     io.imsave(os.path.join(save_dir, 'img_%04d' % (idx_window_) + '_01.tif'), image[1])
    #     io.imsave(os.path.join(save_dir, 'img_%04d' % (idx_window_) + '_02.tif'), image[2])
    #     io.imsave(os.path.join(save_dir, 'img_%04d' % (idx_window_) + '_03.tif'), image[3])

    # figure
    import matplotlib.pyplot as plt
    plt.figure()

    coord_window=idx2coord(idx_window,input_tw.shape[0:3])
    input_tw = input_tw[coord_window]
    image = (input_tw.detach().cpu().numpy() * 255).astype('uint8')
    for idx_frame in range(0, window_size[0]):
        str = '_w%02d' % (idx_window) + '_q%02d' % (idx_query) + '_d%02d' % (idx_depth) + '_f%02d'% (idx_frame)
        image_temp = image[idx_frame]
        image_temp = cv2.cvtColor(image_temp, cv2.COLOR_GRAY2RGB)
        if idx_frame==0:
            coord_query = idx2coord(idx_query, window_size)
            image_temp[coord_query[1], coord_query[2], :] = [255, 0, 0]
        io.imsave(os.path.join(save_dir, 'img' + str + '.tif'), image_temp)
        plt.subplot(5, 4, idx_frame+1)
        plt.imshow(image_temp)
        plt.xticks([]), plt.yticks([])
        plt.axis('off')

    # save attmap
    att_ls=get_local.cache['WindowAttention.attention']
    att=att_ls[idx_depth][idx_window,:,idx_query,:]
    for idx_head in range(0,att.shape[0]):
        att_temp1 = att[idx_head].reshape(window_size)
        att_temp1 = norm(att_temp1)
        for idx_frame in range(0,window_size[0]):
            str = '_w%02d' % (idx_window) + '_q%02d' % (idx_query) + '_d%02d' % (idx_depth) + '_h%02d' % (idx_head) + '_f%02d' % (idx_frame)
            att_temp2=att_temp1[idx_frame]
            io.imsave(os.path.join(save_dir, 'att' + str + '.tif'), att_temp2)
            plt.subplot(5, 4, 4*(idx_head+1)+idx_frame+1)
            plt.imshow(att_temp2)
            plt.xticks([]), plt.yticks([])
            plt.axis('off')

    str = '_w%02d' % (idx_window) + '_q%02d' % (idx_query) + '_d%02d' % (idx_depth)
    # plt.xlabel(['frame_1','frame_2','frame_3','frame_4'])
    # plt.ylabel(['head_1', 'head_2', 'head_3', 'head_4'])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'att'+str+'.png'))
    plt.show()


if __name__ == '__main__':
    input_pth = "/mnt/data1/EMData/Data/1220/test_4.h5"
    scale_factor = 4
    ckpt_pth = "/mnt/data1/EMData/D_Ours/6_VRT/ckpt/VRT_1_23/model_epoch_230.pth"
    save_dir = "/mnt/data1/EMData/Figure/figure1/attmap/VRT_1_23_0"
    os.makedirs(save_dir, exist_ok=True)

    roi = np.s_[70:70 + 64, 56:56 + 16, 1138:1138 + 256]
    idx_depth=3 # depths=[4,4,4,4,4,4,4, 2,2,2,2, 2,2], choose self-attention-only and no-downsample layer in TMSAG
    window_size = [4,8,32]
    # idx_window_ls=[167,  0,94,  8,72,  17,202] # Monolayer, Vesicle, Bilayer
    # idx_query_ls= [84,  133,174,  92,89,  179,114] # Monolayer, Vesicle, Bilayer
    idx_window=167 # img_size=[16,64,256]//window_size=[4,8,32]=256
    idx_query=84 # window_size=[4,8,32]
    get_local.activate()
    input = h5py.File(input_pth, 'r')['raw'][roi]
    model = torch.load(ckpt_pth).eval()
    vis_attmap(model,input,save_dir,idx_depth,window_size,idx_window,idx_query)


