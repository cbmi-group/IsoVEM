import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io
import h5py
from utils import rotate_rand8

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class ImageDataset_train(Dataset):
    '''
    Create training dataset.
    :param image_pth: the path of volume electron microscopy 3d input data.
    :param image_split: the percent of training data in the input data.
    :param subvol_shape: the subvolume size for model processing.
    :param scale_factor: the scale factor of isotropic reconstruction.
    :param is_inpaint: if False, perform normal isovem training, learning isotropic reconstruction task.
                       if True, perform isovem+ training, co-learning the slice inpainting and isotropic reconstruction task.
    :return hr and lr: self-supervised paired data for isotropic reconstruction training.
    :return name: the 3d coordination of current subvolume.
    '''
    def __init__(self, image_pth, image_split, subvol_shape, scale_factor, is_inpaint=False):
        super(ImageDataset_train, self).__init__()

        # load data
        if image_pth.split('.')[-1]=='tif':
            self.image_file = io.imread(image_pth)
        elif image_pth.split('.')[-1] == 'h5':
            self.image_file = np.array(h5py.File(image_pth, 'r')['raw'])
        else:
            raise ValueError(f'Not support the image format of {image_pth}')

        # split train set
        shapey = self.image_file.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file=self.image_file[:,0:shapey_train,:]

        # train settings
        self.subvol_shape = subvol_shape
        self.scale_factor= scale_factor
        self.is_inpaint = is_inpaint

        # dataset size
        self.len=500 # the number of random subvolume sampling per training epoch


    def __getitem__(self, index):
        # random crop
        z = np.random.random_integers(0, self.image_file.shape[0] - self.subvol_shape[0])
        y = np.random.random_integers(0, self.image_file.shape[1] - self.subvol_shape[1])
        x = np.random.random_integers(0, self.image_file.shape[2] - self.subvol_shape[2])
        img = self.image_file[z:z + self.subvol_shape[0], y:y + self.subvol_shape[1], x:x + self.subvol_shape[2]]

        # hr
        hr_transform = transforms.Compose([transforms.ToTensor()])
        img_hr = hr_transform(img.transpose(1, 2, 0).astype('uint8'))
        img_hr = img_hr.unsqueeze(0).unsqueeze(0)
        img_hr = rotate_rand8(img_hr)

        # lr
        down_sample = torch.nn.AvgPool3d(kernel_size=(1, self.scale_factor, 1))
        img_lr = down_sample(img_hr)
        if self.is_inpaint: # random set the slice to be black
            n=np.random.random_integers(0, img_lr.shape[2]-1)
            img_lr[:,:,n,:,:]=0

        img_hr = img_hr.squeeze().unsqueeze(1)
        img_lr = img_lr.squeeze().unsqueeze(1)

        return {"hr": img_hr,"lr": img_lr, "subvol": (z, y, x)}


    def __len__(self):
        return self.len


class ImageDataset_val(Dataset):
    '''
    Create validation dataset
    :param image_pth: the path of volume electron microscopy 3d input data.
    :param image_split: the percent of training data in the input data.
    :param subvol_shape: the subvolume size for model processing.
    :param scale_factor: the scale factor of isotropic reconstruction.
    :param is_inpaint: if False, perform normal isovem training, learning isotropic reconstruction task.
                       if True, perform isovem+ training, co-learning the slice inpainting and isotropic reconstruction task.
    :return hr and lr: self-supervised paired data for isotropic reconstruction training.
    :return name: the 3d coordination of current subvolume.
    '''
    def __init__(self, image_pth, image_split, subvol_shape, scale_factor, is_inpaint=False):
        super(ImageDataset_val, self).__init__()

        # load data
        if image_pth.split('.')[-1]=='tif':
            self.image_file = io.imread(image_pth)
        elif image_pth.split('.')[-1] == 'h5':
            self.image_file = np.array(h5py.File(image_pth, 'r')['raw'])
        else:
            raise ValueError(f'Not support the image format of {image_pth}')

        # split val set
        shapey = self.image_file.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file=self.image_file[:,shapey_train:,:]

        # val settings
        self.subvol_shape = subvol_shape
        self.scale_factor= scale_factor
        self.is_inpaint = is_inpaint

        # generate subvolume coordinates in order, different from random cropping in training dataset
        import math
        import itertools
        z_crop_num = math.ceil(self.image_file.shape[0] / subvol_shape[0])
        y_crop_num = math.ceil(self.image_file.shape[1] / subvol_shape[1])
        x_crop_num = math.ceil(self.image_file.shape[2] / subvol_shape[2])
        self.coords= []
        for z, y, x in itertools.product(range(z_crop_num), range(y_crop_num), range(x_crop_num)):
            z_coord = z * subvol_shape[0] if z * subvol_shape[0] < self.image_file.shape[0] - subvol_shape[0] else self.image_file.shape[0] - subvol_shape[0]
            y_coord = y * subvol_shape[1] if y * subvol_shape[1] < self.image_file.shape[1] - subvol_shape[1] else self.image_file.shape[1] - subvol_shape[1]
            x_coord = x * subvol_shape[2] if x * subvol_shape[2] < self.image_file.shape[2] - subvol_shape[2] else self.image_file.shape[2] - subvol_shape[2]
            self.coords.append((z_coord,y_coord,x_coord))
        self.len = len(self.coords) # the number of cropped subvolumes


    def __getitem__(self, index):
        # crop out subvolume based on coordinate
        z,y,x=self.coords[index]
        img = self.image_file[z:z + self.subvol_shape[0], y:y + self.subvol_shape[1], x:x + self.subvol_shape[2]]

        # hr
        hr_transform = transforms.Compose([transforms.ToTensor()])
        img_hr = hr_transform(img.transpose(1, 2, 0).astype('uint8'))
        img_hr = img_hr.unsqueeze(0).unsqueeze(0)

        # lr
        down_sample = torch.nn.AvgPool3d(kernel_size=(1, self.scale_factor, 1))
        img_lr = down_sample(img_hr)
        if self.is_inpaint: # random set the slice to be black
            n=np.random.random_integers(0, img_lr.shape[2]-1)
            img_lr[:,:,n,:,:]=0

        img_hr = img_hr.squeeze().unsqueeze(1)
        img_lr = img_lr.squeeze().unsqueeze(1)

        return {"hr": img_hr, "lr": img_lr, "subvol": (z, y, x)}

    def __len__(self):
        return self.len