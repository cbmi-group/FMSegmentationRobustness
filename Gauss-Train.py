# -*- coding: UTF-8 -*-

from datetime import datetime
import argparse
import logging
import os
import sys

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "models"))
from models.FCN import FCNs,VGGNet
from models.UNet_5 import UNet
from models.segnet import SegNet
from models.simple_unet import simple_unet
from models.UNet_3 import UNet_3
from models.DeepLab.deeplab_v3 import DeepLab
from models.PSPNet.PSPNet import PSPNet
from models.ICNet.ICNet import ICNet
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from eval_noise import eval_net
import time
from dataset_random_noise import BasicDataset
from torch.utils.data import DataLoader, random_split
from losses import *
from Parameter import get_args
from train_attack import *

args = get_args()
print('GPU id:',args.gpu)
os.environ['CUDA_VISIBLE_DEVICE'] = '0,1'
torch.cuda.set_device(args.gpu)

def train(model,
          device,
          epochs,
          batch_size,
          lr,
          direction,
          attack
        ):

    # vis = visdom.Visdom()

    train = BasicDataset(train_dir, scale=1, direction=direction, norm=args.norm)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # num_wokers表示并行计算的CPU数量，比如np.random的随机数是一次性产生的，并且将产生的随机数统一复制分配到8个CPU待用，以后每个epoch都调用固定的随机数，如果data没有shuffle则那同一
    # 图像每轮 epoch对应相同的随机数，如果data shuffle则固定的随机数对应不固定的图像序列，因此同一图像每轮 epoch 都会分配不同的随机数
    val = BasicDataset(val_dir, scale=1, direction=direction, norm=args.norm)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    n_train = len(train)
    n_val = len(val)

    temp = train_dir[5:]
    writer = SummaryWriter(comment=f'_{model_name}_{lr}_{args.norm}_{temp}_save2_{dir_checkpoint}')
    global_step = 0
    logging.info(f'''Starting training:
            Data preprocess  {args.norm}
            Model name:      {model_name}
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {device.type}
            training dir:    {train_dir}
            val dir:    {val_dir}
            saving weights to:  {dir_checkpoint}
            GPU id:          {args.gpu}
            direction:       {direction}
            PGD iter:  {args.k}
            PGD epsilon:   {args.epsilon}
            PGD step:  {args.alpha}
        ''')
    # set optimizer
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(),lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.output_c > 1 else 'max', factor=0.5,patience=10)

    if model_name == 'ICNet':
        logging.info(f'Loss_function: ICNetLoss')
        criterion = ICNetLoss()
    elif model_name == 'PSPNet':
        logging.info(f'Loss_function: PSPNetLoss')
        criterion = PSPNetLoss()
    else:
        if model.output_c == 1:
            logging.info(f'Loss_function: BCEWithLogitsLoss')
            criterion = nn.BCEWithLogitsLoss() # output维度(batch,1,h,w),target维度(batch,1,h,w)
        else:
            logging.info(f'Loss_function: CrossEntropyLoss')
            criterion = nn.CrossEntropyLoss()  # output维度(batch,2,h,w),target维度(batch,h,w)
    # criterion = nn.CrossEntropyLoss()  # 将nn.LogSoftmax()（归一化）和nn.NLLLoss()整合成一个类
    # cls_criterion = nn.BCEWithLogitsLoss()  # 将 sigmoid（归一化）和与 BCELoss整合成一个类。
    # start timing
    criterion = criterion.to(device=device)
    prev_score = float('-inf')

    for epoch in range(epochs):
        print('+'*10, epoch)
        epoch_loss = 0
        model.train()
        start_time = time.time()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:  # with tqdm as pbar 手动控制进度条更新步长
            for batch in train_loader:

                # print('/' * 10, 'batch')
                global_step += 1
                clean = batch['clean']
                true_masks = batch['mask']
                noisy = batch['noisy']
                clean = clean.to(device=device, dtype=torch.float32)
                noisy = noisy.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                # std = std.to(device=device, dtype=torch.float32)
                # std = torch.mean(std)
                # print('masks_pred size:',masks_pred.size())
                # print('ture mask size:', std)

                output = model(clean)
                loss = criterion(output, true_masks)

                output_noise = model(noisy)
                loss_noise = criterion(output_noise, true_masks)
                loss = (loss + loss_noise) / 2

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})  # 设置进度条右边显示信息，t.set_description：设置左边显示信息

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(fcn_model.parameters(), 0.1)
                optimizer.step()

                pbar.update(clean.shape[0])   # 以一个batch为步长更新进度条

        # validation
        if (epoch+1) % 1 == 0:

            val_score = eval_net(model, val_loader, device)

            scheduler.step()  # 学习率衰减

            llr = optimizer.param_groups[0]['lr']

            logging.info(f'Model:{model_name} // learning rate: {llr}')
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            logging.info(f'Loss: {epoch_loss/batch_size} ')
            logging.info('Validation mIoU: {}'.format(val_score))
            writer.add_scalar('mIoU/valid', val_score, global_step)

            writer.add_images('images', noisy, global_step)

            if model.output_c == 1:
                writer.add_images('masks/true', true_masks, global_step)
                if type(output) == tuple:
                    writer.add_images('masks/pred', torch.sigmoid(output[0]) > 0.5, global_step)
                else:
                    writer.add_images('masks/pred', torch.sigmoid(output) > 0.5, global_step)
            else:
                writer.add_images('masks/true', true_masks, global_step)
                if type(output) == tuple:
                    writer.add_images('masks/pred', F.softmax(output[0], dim=1)[:,1:2,:,:] > 0.5, global_step)
                else:
                    writer.add_images('masks/pred', F.softmax(output, dim=1)[:, 1:2, :, :] > 0.5, global_step)

        # save the best model weight
        is_better = val_score > prev_score
        if is_better:
            prev_score = val_score
            torch.save(model.state_dict(), dir_checkpoint + 'model_best.pth')
            logging.info(f'Best model saved !')
        # save model weight per 10 epoch
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

            # logging.info(f'Checkpoint {epoch + 1} saved !')
        if epoch  == epochs - 1:
            logging.info(f'''Starting training:
                        Data preprocess  {args.norm}
                        Model name:      {model_name}
                        Epochs:          {epochs}
                        Batch size:      {batch_size}
                        Learning rate:   {lr}
                        Training size:   {n_train}
                        Validation size: {n_val}
                        Device:          {device.type}
                        training dir:    {train_dir}
                        val dir:    {val_dir}
                        saving weights to:  {dir_checkpoint}
                        GPU id:          {args.gpu}
                        direction:       {direction}
                        PGD iter:  {args.k}
                        PGD epsilon:   {args.epsilon}
                        PGD step:  {args.alpha}
                    ''')
        end_time = time.time()
        delta = end_time - start_time
        logging.info(f'Using time: {delta} seconds')
    writer.close()

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    model_name = args.model
    name = args.train_dir[0]

    train_dir = 'data/' + name  # input data directory
    name = args.val_dir[0]
    val_dir = 'data/' + name
    dir_checkpoint = 'checkpoints/' + model_name + '/' + args.save_pth  # save weights to...
    os.makedirs(dir_checkpoint, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # load model
    if model_name == 'FCNs':
        vgg_model = VGGNet(requires_grad=True)
        model = eval(model_name)(pretrained_net=vgg_model)
    else:
        model = eval(model_name)()
    # print(model)
    model = model.to(device)
    attack = PGD_attack(model,
                        args.epsilon,
                        args.alpha,
                        max_iters=args.k,
                        _type=args.perturbation_type)
    train(model=model,
          epochs=args.epochs,
          batch_size=args.batchsize,
          lr=args.lr,
          device=device,
          direction=args.direction,
          attack=attack)
