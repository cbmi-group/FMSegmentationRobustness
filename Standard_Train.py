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
from eval import eval_net
import time
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from losses import *
from Parameter import get_args

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
        ):

    train = BasicDataset(train_dir, scale=1, direction=direction, norm=args.norm)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
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
        ''')
    # set optimizer
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(),lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
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
            criterion = nn.BCEWithLogitsLoss()
        else:
            logging.info(f'Loss_function: CrossEntropyLoss')
            criterion = nn.CrossEntropyLoss()

    # start timing
    prev_score = float('-inf')

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        start_time = time.time()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                global_step += 1
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                # print('masks_pred size:',masks_pred.size())
                # print('ture mask size:', true_masks.size())
                masks_pred = model(imgs)

                loss = criterion(masks_pred, true_masks)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(fcn_model.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
        # validation
        if (epoch+1) % 1 == 0:

            val_score = eval_net(model, val_loader, device)

            scheduler.step()

            llr = optimizer.param_groups[0]['lr']

            logging.info(f'Model:{model_name} // learning rate: {llr}')
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            logging.info('Validation mIoU: {}'.format(val_score))
            writer.add_scalar('mIoU/valid', val_score, global_step)

            writer.add_images('images', imgs, global_step)

            if model.output_c == 1:
                writer.add_images('masks/true', true_masks, global_step)
                if type(masks_pred) == tuple:
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred[0]) > 0.5, global_step)
                else:
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
            else:
                writer.add_images('masks/true', true_masks, global_step)
                if type(masks_pred) == tuple:
                    writer.add_images('masks/pred', F.softmax(masks_pred[0], dim=1)[:,1:2,:,:] > 0.5, global_step)
                else:
                    writer.add_images('masks/pred', F.softmax(masks_pred, dim=1)[:, 1:2, :, :] > 0.5, global_step)

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
                    ''')
        end_time = time.time()
        delta = end_time - start_time
        logging.info(f'Using time: {delta} seconds')
    writer.close()

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    train_dir = args.train_dir  # training data directory
    val_dir = args.val_dir  # valid data directory

    model_name = args.model
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

    train(model=model,
          epochs=args.epochs,
          batch_size=args.batchsize,
          lr=args.lr,
          device=device,
          direction=args.direction)
