
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
from PIL import Image, ImageFile
from torchvision import transforms as torch_transform
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import pylab
import matplotlib.patches as mpatches
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange


#TO DOS --- Work out what the default clipping value and alpha value should be 
# Implement each of the four attacks

def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)

    elif _type == 'l2':
        dist = (x - original_x)

        dist = dist.view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)
        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
        # dist = F.normalize(dist, p=2, dim=1)
        dist = dist / dist_norm
        dist *= epsilon
        dist = dist.view(x.shape)
        x = (original_x + dist) * mask.float() + x * (1 - mask.float())

    else:
        raise NotImplementedError

    return x
class Attack():
    """
    FGSM class containing all attacks and miscellaneous functions 
    """

    def __init__(self,model, loss, device):
        """ Creates an instance of the FGSM class.
        """
        self.model = model
        self.loss = loss
        self.device = device
        self.mask_type = torch.float32 if self.model.output_c == 1 else torch.long
        #set the model to evaluation mode for FGSM attacks
        self.model.eval()

    def shuffle_zlq(self, img):
        img = img[:,:,torch.randperm(img.size(2)),:]
        img = img[:,:,:,torch.randperm(img.size(3))]
        return img

    def FGSM(self,img, labels, epsilons):

        if epsilons == []:
            raise Exception("alphas must be a non empty list")

        pred = self.model(img)
        l = self.loss(pred,labels)
        img.retain_grad()
        z = torch.sum(l)
        z.backward(retain_graph=True)
        im_grad = img.grad
        noise_list = [eps*torch.sign(im_grad) for eps in epsilons]
        # [print('*'*5,noise.shape) for noise in noise_list]
        # noise_list = [self.shuffle_zlq(noise) for noise in noise_list]  # 打算对抗扰动的结构，使其退化为噪声攻击
        adv_list = [img + noise for noise in noise_list]
        return adv_list, noise_list

    def PGD(self, original_images, labels, epsilons, alpha, max_iters = 10,  random_start = True):
        adv_list = []
        for epsilon in epsilons:
            # print('+'*10,epsilon)
            if random_start:
                rand_perturb = torch.FloatTensor(original_images.shape).uniform_(-epsilon, epsilon)
                # rand_perturb = tensor2cuda(rand_perturb)
                rand_perturb = rand_perturb.to(device=self.device, dtype=torch.float32)
                x = original_images + rand_perturb
                # x.clamp_(self.min_val, self.max_val)
            else:
                x = original_images.clone()

            x.retain_grad()

            with torch.enable_grad():
                for _iter in range(max_iters):

                    labels = labels.to(device=self.device, dtype=self.mask_type)
                    x = x.to(device=self.device, dtype=torch.float32)
                    outputs = self.model(x)
                    L = self.loss (outputs, labels)
                    grads = torch.autograd.grad(L, x, grad_outputs=None, only_inputs=True)[0]
                    x.data += alpha * torch.sign(grads.data)
                    # the adversaries' pixel value should within max_x and min_x due
                    # to the l_infinity / l2 restriction
                    x = project(x, original_images, epsilon)  # 将 x 映射到 epsilon 范围内
                    # the adversaries' value should be valid pixel value
                    # x.clamp_(self.min_val, self.max_val)
            # self.model.train()
            adv_list.append(x)

        return adv_list

    def I_FGSM(self, img, labels, epsilons, alpha, iters):

        if epsilons == []:
            raise Exception("alphas must be a non empty list")

        noise_list = []
        adv_list = []

        for epsilon, it in zip(epsilons, iters):
            # tbar = trange(it)
            it = 1000
            epsilon = epsilons[4]
            x = img.clone()
            x.retain_grad()
            with torch.enable_grad():
                for i in range(it):

                    labels = labels.to(device=self.device, dtype=self.mask_type)
                    x = x.to(device=self.device, dtype=torch.float32)
                    if self.model.output_c == 2:
                        labels = torch.squeeze(labels, 1)
                    outputs = self.model(x)
                    L = self.loss(outputs, labels)
                    grads = torch.autograd.grad(L, x, grad_outputs=None, only_inputs=True)[0]
                    print('i = ', i+1, 'loss: ', L.data)
                    x.data += alpha * torch.sign(grads.data)
                    x = project(x, img, epsilon)
                    # tbar.set_description('Iteration: {}/{} of iterated-FGSMI attack'.format((i+1), it))
            e
            adv_list.append(x)

        return adv_list



