from dataset import *
from metric import *
import numpy as np

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def rotate_8(imgs_hr):
    '''
    8 kinds of orthogonal rotations, keeping anisotropic axis unchanged.
    imgs_hr should be a 3d data with (Z,Y,X) order.
    utils for test-time-augmentation.
    '''
    alpha_ls = [-2,0]
    beta_ls = [-2]
    gamma_ls = [-2, -1,0,1]

    imgs_hr_ls=[]
    idx=-1
    for alpha in alpha_ls:
        for beta in beta_ls:
            for gamma in gamma_ls:
                idx+=1
                imgs_hr_copy = imgs_hr.clone()
                imgs_hr_copy=torch.rot90(imgs_hr_copy, 2+alpha, [2, 4])
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + beta, [2, 3])
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + gamma, [3, 4])
                imgs_hr_ls.append(imgs_hr_copy)
    return imgs_hr_ls


def anti_rotate_8(imgs_hr_ls,image_shape=(128,128,128)):
    '''
    anti-rotation for 8 kinds of orthogonal rotations.
    images in imgs_hr_ls should be 3d data with (Z,Y,X) order.
    utils for test-time-augmentation.
    '''
    alpha_ls = [-2, 0]
    beta_ls = [-2]
    gamma_ls = [-2, -1, 0, 1]

    idx=-1
    imgs_anti_rot =[]
    res=torch.zeros((1,1,image_shape[0],image_shape[1],image_shape[2])).type(Tensor)
    for alpha in alpha_ls:
        for beta in beta_ls:
            for gamma in gamma_ls:
                idx+=1
                imgs_hr_copy = imgs_hr_ls[idx]
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 4-(2 +gamma), [3, 4])
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 4-(2 + beta), [2, 3])
                imgs_hr_copy=torch.rot90(imgs_hr_copy, 4-(2+alpha), [2, 4])
                imgs_anti_rot.append(imgs_hr_copy)
                res+=imgs_hr_copy

    return imgs_anti_rot,res/8


def blend_X(img1, img2, overlap):
    '''2d stitch along X axis, utils for 3d subvolume stitching.'''
    b, a = img1.shape
    d, c = img2.shape
    res = np.zeros([b, a+c - overlap]).astype(np.float16)

    for i in range(a+c - overlap):
        if i < a-overlap:
            weight1 = 1
            res[:, i] = weight1 * img1[:, i]
        if a-overlap <= i < a:
            weight1 = -i + a
            weight2 = i - a + overlap
            weight1 = weight1 / overlap
            weight2 = weight2 / overlap
            res[:, i] = weight1 * img1[:, i] + weight2 * img2[:, i - a + overlap]
        if i >= a:
            weight2 = 1
            res[:, i] = weight2 * img2[:, i-a+overlap]

    return res

def blend_Y(img1, img2, overlap):
    '''2d stitch along Y axis, utils for 3d subvolume stitching.'''
    b, a = img1.shape
    d, c = img2.shape
    res = np.zeros([b+d - overlap, a]).astype(np.float16)
    for i in range(b+d - overlap):
        if i < b-overlap:
            weight1 = 1
            res[i, :] = weight1 * img1[i, :]
        if b-overlap <= i < b:
            weight1 = -i + b
            weight2 = i - b + overlap
            weight1 = weight1 / overlap
            weight2 = weight2 / overlap
            res[i, :] = weight1 * img1[i, :] + weight2 * img2[i - b + overlap, :]
        if i >= b:
            weight2 = 1
            res[i, :] = weight2 * img2[i-b+overlap, :]

    return res

def blend3D_X(img1, img2, overlap=16):
    '''3d stitch along X axis, utils for 3d subvolume stitching.'''
    c, b, a = img1.shape
    z, y, x = img2.shape
    res = np.zeros([c, b, a+x - overlap]).astype(np.float16)
    for j in range(b):
        res[:, j, :]=blend_X(img1[:, j, :], img2[:, j, :], overlap=overlap)
    return res


def blend3D_Y(img1, img2, overlap=16):
    '''3d stitch along Y axis, utils for 3d subvolume stitching.'''
    c, b, a = img1.shape
    z, y, x = img2.shape
    res = np.zeros([c, b + y - overlap, a]).astype(np.float16)
    for j in range(a):
        res[:, :, j] = blend_X(img1[:, :, j], img2[:, :, j], overlap=overlap)
    return res


def blend3D_Z(img1, img2, overlap=16):
    '''3d stitch along Z axis, utils for 3d subvolume stitching.'''
    c, b, a = img1.shape
    z, y, x = img2.shape
    res = np.zeros([c + z - overlap, b, a]).astype(np.float16)
    for j in range(b):
        res[:, j, :] = blend_Y(img1[:, j, :], img2[:, j, :], overlap=overlap)
    return res

