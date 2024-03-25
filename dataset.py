import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def rotate_rand8(imgs_hr):
    '''
    random orthogonal rotations, keeping anisotropic axis unchanged.
    imgs_hr should be a 3d data with (Z,Y,X) order.
    '''
    alpha = random.choice([-2, 0])
    beta = random.choice([-2])
    gamma = random.choice([-2, -1, 0, 1])

    imgs_hr_copy = imgs_hr.clone()
    imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + alpha, [2, 4])
    imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + beta, [2, 3])
    imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + gamma, [3, 4])

    return imgs_hr_copy


class ImageDataset_train(Dataset):
    '''
    Create training dataset
    :param image_h5: the path of volume electron microscopy 3d data.
    :param image_shape: the subvolume size for model processing.
    :param scale_factor: the scale factor of isotropic reconstruction.
    :param is_gt: if image_h5 is an isotropic data, set True, and simulated anisotropy degradation is performed on this data.
                  if image_h5 is an anisotropic data, set False.
    :param is_inpaint: if False, perform normal isovem training, learning isotropic reconstruction task.
                       if True, perform isovem+ training, co-learning the slice inpainting and isotropic reconstruction task.
    :return gt: if image_h5 is an isotropic data, gt is image_h5.
                if image_h5 is an anisotropic data, gt is zeros.
    :return hr and lr: self-supervised paired data for isotropic reconstruction training.
    :return name: the 3d coordination of current subvolume.
    '''
    def __init__(self, image_h5,image_shape=(128,128,128),scale_factor=8,is_gt=True,is_inpaint=False):
        super(ImageDataset_train, self).__init__()
        self.h5file =io.imread(image_h5) # (Z,Y,X)
        self.image_shape = image_shape # (Z,Y,X)
        self.scale_factor=scale_factor
        self.is_gt = is_gt
        self.is_inpaint = is_inpaint
        self.hr_transform = transforms.Compose([transforms.ToTensor()])
        self.len=500 # the number of random subvolume sampling per training epoch

    def __getitem__(self, index):
        # random crop
        z = np.random.random_integers(0, self.h5file.shape[0] - self.image_shape[0])
        y = np.random.random_integers(0, self.h5file.shape[1] - self.image_shape[1])
        x = np.random.random_integers(0, self.h5file.shape[2] - self.image_shape[2])
        img = self.h5file[z:z + self.image_shape[0], y:y + self.image_shape[1], x:x + self.image_shape[2]]

        # hr
        img_hr = self.hr_transform(img.transpose(1, 2, 0).astype('uint8'))  # YXZ->ZYX
        img_hr = img_hr.unsqueeze(0).unsqueeze(0)
        # random rotate
        img_hr = rotate_rand8(img_hr)

        # gt
        img_gt=torch.empty_like(img_hr)
        if self.is_gt: # perform simulated anisotropy degradation
            img_gt=img_hr
            img_gt=img_gt.squeeze().unsqueeze(1)
            down_sample = torch.nn.AvgPool3d(kernel_size=(self.scale_factor, 1, 1))
            img_hr = down_sample(img_hr)

        # lr
        down_sample = torch.nn.AvgPool3d(kernel_size=(1, self.scale_factor, 1))
        img_lr = down_sample(img_hr)
        if self.is_inpaint: # random set the slice to be black
            n=np.random.random_integers(0, img_lr.shape[2]-1)
            img_lr[:,:,n,:,:]=0

        img_hr = img_hr.squeeze().unsqueeze(1)
        img_lr = img_lr.squeeze().unsqueeze(1)

        return {"gt":img_gt,"hr": img_hr,"lr": img_lr, "name": (z, y, x)}


    def __len__(self):
        return self.len


class ImageDataset_val(Dataset):
    '''
    Create validation dataset
    :param image_h5: the path of volume electron microscopy 3d data.
    :param image_shape: the subvolume size for model processing.
    :param scale_factor: the scale factor of isotropic reconstruction.
    :param is_gt: if image_h5 is an isotropic data, set True, and simulated anisotropy degradation is performed on this data.
                  if image_h5 is an anisotropic data, set False.
    :param is_inpaint: if False, perform normal isovem training, learning isotropic reconstruction task.
                       if True, perform isovem+ training, co-learning the slice inpainting and isotropic reconstruction task.
    :return gt: if image_h5 is an isotropic data, gt is image_h5.
                if image_h5 is an anisotropic data, gt is zeros.
    :return hr and lr: self-supervised paired data for isotropic reconstruction training.
    :return name: the 3d coordination of current subvolume.
    '''
    def __init__(self, image_h5, image_shape=(128, 128, 128), scale_factor=8, is_gt=True,is_inpaint=False):
        super(ImageDataset_val, self).__init__()
        self.h5file = io.imread(image_h5) # (Z,Y,X)
        self.image_shape = image_shape  # (Z,Y,X)
        self.scale_factor = scale_factor
        self.is_gt = is_gt
        self.is_inpaint = is_inpaint
        self.hr_transform = transforms.Compose([transforms.ToTensor()])

        # generate subvolume coordinates in order, different from random cropping in training dataset
        import math
        import itertools
        z_crop_num = math.ceil(self.h5file.shape[0] / image_shape[0])
        y_crop_num = math.ceil(self.h5file.shape[1] / image_shape[1])
        x_crop_num = math.ceil(self.h5file.shape[2] / image_shape[2])
        self.coords= []
        for z, y, x in itertools.product(range(z_crop_num), range(y_crop_num), range(x_crop_num)):
            z_coord = z * image_shape[0] if z * image_shape[0] < self.h5file.shape[0] - image_shape[0] else self.h5file.shape[0] - image_shape[0]
            y_coord = y * image_shape[1] if y * image_shape[1] < self.h5file.shape[1] - image_shape[1] else self.h5file.shape[1] - image_shape[1]
            x_coord = x * image_shape[2] if x * image_shape[2] < self.h5file.shape[2] - image_shape[2] else self.h5file.shape[2] - image_shape[2]
            self.coords.append((z_coord,y_coord,x_coord))
        self.len = len(self.coords) # the number of cropped subvolume

    def __getitem__(self, index):
        # crop out subvolume based on coordinate
        z,y,x=self.coords[index]
        img = self.h5file[z:z + self.image_shape[0], y:y + self.image_shape[1], x:x + self.image_shape[2]]
        img_hr = self.hr_transform(img.transpose(1, 2, 0).astype('uint8'))  # YXZ->ZYX
        img_hr = img_hr.unsqueeze(0).unsqueeze(0)

        # gt
        img_gt = torch.empty_like(img_hr)
        if self.is_gt: # perform simulated anisotropy degradation
            img_gt = img_hr
            img_gt = img_gt.squeeze().unsqueeze(1)
            down_sample = torch.nn.AvgPool3d(kernel_size=(self.scale_factor, 1, 1))
            img_hr = down_sample(img_hr)

        # lr
        down_sample = torch.nn.AvgPool3d(kernel_size=(1, self.scale_factor, 1))
        img_lr = down_sample(img_hr)
        if self.is_inpaint: # random set the slice to be black
            n=np.random.random_integers(0, img_lr.shape[2]-1)
            img_lr[:,:,n,:,:]=0

        img_hr = img_hr.squeeze().unsqueeze(1)
        img_lr = img_lr.squeeze().unsqueeze(1)

        return {"gt": img_gt, "hr": img_hr, "lr": img_lr, "name": (z, y, x)}

    def __len__(self):
        return self.len