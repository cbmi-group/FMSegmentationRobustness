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
        self.scale = scale
        self.direction = direction
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        dirs = os.listdir(root)
        self.imgs = [os.path.join(root,img) for img in dirs]

        logging.info(f'Creating dataset with {len(self.imgs)} examples')

    def __len__(self):
        return len(self.imgs)

    @classmethod
    def preprocess(cls, pil_img, norm,img_path,scale=1):
        w, h = pil_img.shape
        newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        img_nd = np.asarray(pil_img,dtype = np.float32)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        temp1 = np.zeros((3,w,h))
        temp2 = np.zeros((3,w,h))
        clean = img_nd.transpose((2, 0, 1))

        # img_trans = img_trans.astype('float64')
        # print('img max:',np.max(img_trans))
        # mean = np.random.randint(30,50)

        mean = 0
        sigma = np.random.randint(8)
        # if img_path == 'data/mito/612_SNR_8/530.png':
        # print(f'{img_path}, {sigma}')
        noise = np.random.normal(mean, sigma, size = clean.shape)
        noise = np.float32(noise)

        img_trans = clean + noise
        img_trans[img_trans<=0] = 0
        img_trans[img_trans>=255] = 255

        if norm == 'std':
            img_trans = (img_trans - np.mean(img_trans)) / np.std(img_trans)
            clean = (clean - np.mean(clean)) / np.std(clean)
        else:
            img_trans = img_trans / 255
            clean = clean / 255

        temp1[0, :, :] = img_trans
        temp1[1, :, :] = img_trans
        temp1[2, :, :] = img_trans
        img_trans = temp1
        temp2[0, :, :] = clean
        temp2[1, :, :] = clean
        temp2[2, :, :] = clean
        clean = temp2
        # if img_trans.max() > 1:
        #     img_trans = img_trans / 255

        return clean, img_trans

    def __getitem__(self, index):

        img_path = self.imgs[index]

        #assert len(mask_file) == 1, \
            #f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        #assert len(img_file) == 1, \
        #    f'Either no image or multiple images found for the ID {idx}: {img_file}'

        # print('img_file:',img_file[0])
        # print('mask_file:',mask_file[0])
        temp = Image.open(img_path)  # [w,h]

        temp = np.asarray(temp)
        h, w = temp.shape
        if self.direction == 'AtoB':
            image = np.double(temp[:, 0:h])
            mask = np.double(temp[:, h:2 * h])
        else:
            mask = np.double(temp[:, 0:h])
            image = np.double(temp[:, h:2 * h])

        # img1 = img.load()
        # mask1 = mask.load()
        #print('mask shape:',mask.shape)
        #print('image shape:',image.shape)
        norm = 'std'

        assert image.shape == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        clean, noisy = self.preprocess(image, norm, img_path,self.scale)
        # mask = self.preprocess(mask, self.scale)
        mask = mask / 255.
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        mask = mask.transpose((2, 0, 1))
        return {'noisy': torch.from_numpy(noisy), 'clean': torch.from_numpy(clean), 'mask': torch.from_numpy(mask)}
