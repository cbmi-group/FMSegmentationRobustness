from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os


class BasicDataset(Dataset):
    def __init__(self, root, scale=1, direction='AtoB',norm='std'):

        # self.imgs_dir = imgs_dir
        self.norm = norm
        self.scale = scale
        self.direction = direction
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        dirs = os.listdir(root)
        self.imgs = [os.path.join(root,img) for img in dirs]

        logging.info(f'Creating dataset with {len(self.imgs)} examples')

    def __len__(self):
        return len(self.imgs)

    @classmethod
    def preprocess(cls, pil_img, norm):
        w, h = pil_img.shape
        scale = 1
        newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        temp1 = np.zeros((3,w,h))
        img_trans = img_nd.transpose((2, 0, 1))

        if norm == 'std':
            # print('... Standar_Norm ...')
            img_trans = (img_trans - np.mean(img_trans)) / np.std(img_trans)
        else:
            # print('... 255_Norm ...')
            img_trans = img_trans / 255

        if img_trans.shape[0] == 1 :
            temp1[0, :, :] = img_trans
            temp1[1, :, :] = img_trans
            temp1[2, :, :] = img_trans
        else:
            temp1 = img_trans

        return temp1

    def __getitem__(self, index):

        img_path = self.imgs[index]

        temp = Image.open(img_path)  # [w,h]
        temp = np.asarray(temp)
        h, w = temp.shape
        # print('*'*10,h,w)
        if self.direction == 'AtoB':
            image = np.double(temp[:, 0:h])
            mask = np.double(temp[:, h:2 * h])
        else:
            mask = np.double(temp[:, 0:h])
            image = np.double(temp[:, h:2 * h])

        # img1 = img.load()
        # mask1 = mask.load()
        # print('mask shape:',mask.shape)
        # print('image shape:',image.shape)

        assert image.shape == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        image = self.preprocess(image, self.norm)
        # mask = self.preprocess(mask, self.scale)
        mask = mask / 255.
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
            mask = mask.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'mask': torch.from_numpy(mask)}
