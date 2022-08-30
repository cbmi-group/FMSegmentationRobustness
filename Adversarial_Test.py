import argparse
import logging
import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from models.FCN import FCNs,VGGNet
from models.UNet_5 import UNet
from models.UNet_3 import UNet_3
from models.segnet import SegNet
from models.simple_unet import simple_unet
from models.DeepLab.deeplab_v3 import DeepLab
from models.PSPNet.PSPNet import PSPNet
from models.ICNet.ICNet import ICNet
from test_attack import Attack
from dataset import BasicDataset
from PIL import Image
from torch.autograd import Variable
# import torch.nn.functional as F

# from feature_function import FeatureVisualization

# logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_adv(net,
                full_img,
                device,
                loss,
                targets):
    # print('*********',np.mean(full_img))

    # if normalization
    if args.norm == 'std':
        mean = np.mean(full_img)
        std = np.std(full_img)
    else:
        mean = 0
        std = 255

    img = BasicDataset.preprocess(full_img, args.norm)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    img = Variable(img,requires_grad=True)

    targets = torch.from_numpy(targets).unsqueeze(0).unsqueeze(0)
    targets = targets.to(device=device, dtype=torch.float32)

    # with torch.no_grad():

    attack = Attack(net, loss, device)
    # output = net(img)

    epsilons = [0, 1, 2, 4, 8, 16, 32]
    its = [min(int(epsilon + 4), int(1.25 * epsilon)) for epsilon in epsilons]
    its = [1 if i == 0 else i for i in its]
    epsilons = np.array(epsilons)
    epsilons = epsilons / std

    if args.attack == 'PGD':
        alpha = 2 / std
        l_adv_img_tens = attack.PGD(img, targets, epsilons, alpha)
    elif args.attack == 'FGSM':
        l_adv_img_tens, _ = attack.FGSM(img, targets, epsilons)
    elif args.attack == 'I-FGSM':
        alpha = 1 / std
        l_adv_img_tens = attack.I_FGSM(img, targets, epsilons, alpha, its)

    with torch.no_grad():
        l_adv_out = [net(adv_img_tens) for adv_img_tens in l_adv_img_tens]
        # l_adv_loss = [out_mask(adv_out) for adv_out in l_adv_out]

    l_adv_pred = [out_mask(adv_out) for adv_out in l_adv_out]

    ladv_img_tmp = [adv_img_tens.squeeze().cpu().detach().numpy() for adv_img_tens in l_adv_img_tens]
    ladv_img_tmp = [np.transpose(adv_img_tmp, axes=[1, 2, 0]) for adv_img_tmp in ladv_img_tmp]
    l_adv_img_tmp = [np.uint8(np.clip(np.round(adv_img_tmp * std + mean), 0, 255)) for adv_img_tmp in ladv_img_tmp]
    # l_adv_img_tmp = [np.round(adv_img_tmp * std + mean) for adv_img_tmp in ladv_img_tmp]

    return l_adv_img_tmp, l_adv_pred


def out_mask(output):
    if output.shape[1] > 1:
        probs = F.softmax(output, dim=1)
        probs = torch.argmax(probs, dim=1, keepdim=True)
        # probs = probs[:, 1:2, :, :]
    else:
        probs = torch.sigmoid(output)
    full_mask = probs.squeeze().cpu().numpy()

    return full_mask > 0.5

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='FCNs', choices=['FCNs','UNet','UNet_3', 'simple_unet',
                                                                      'SegNet','DeepLab','PSPNet','ICNet'],)
    parser.add_argument('--load_pth', type=str,default='trained_on_ER_594_SNR_8/',
                        help='model weights directory')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--mask-threshold', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--attack', type=str, default='FGSM',choices=['PGD','FGSM','I-FGSM'])
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to be used. Such as -gpu 0')
    parser.add_argument('--direction', type=str, default='AtoB',
                        help='AtoB or BtoA')
    parser.add_argument('--norm', type=str, default='std',
                        help='which normalization...')
    parser.add_argument('--test_dir', type=str, default='data/mito/test/')
    parser.add_argument('--best', type=bool, default=True,help='Whether to load the best model')

    return parser.parse_args()

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def IoU(input, target):
    eps = 0.00001
    input = np.float32(input)
    inter = np.dot(input.flatten(), target.flatten())
    # print('input:', input[0,0:10])
    # print('target:', target.dtype)
    union = np.sum(input) + np.sum(target) - inter
    t = inter / (union + eps)

    return t

if __name__ == "__main__":

    args = get_args()
    print('GPU id:', args.gpu)
    torch.cuda.set_device(args.gpu)
    model_name = args.model

    # load model network
    if model_name == 'FCNs':
        vgg_model = VGGNet(requires_grad=True)
        model = eval(model_name)(pretrained_net=vgg_model)
    else:     model = eval(model_name)()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)

    # initialize loss function
    if model.output_c == 1:
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.CrossEntropyLoss()
    loss = loss.to(device=device)

    # load model weights
    if args.best == 1:
        load_path = 'checkpoints/' + model_name + '/' + args.load_pth + 'model_best.pth'
    else:
        load_path = 'checkpoints/' + model_name + '/' + args.load_pth + 'CP_epoch' + str(args.epochs) + '.pth'
        # load_path = '../Corrupt_Robust/checkpoints/UNet/mito_ST_3lr_std/' + 'CP_epoch' + str(args.epochs) + '.pth'
    model.load_state_dict(torch.load(load_path, map_location=device),strict=False)
    logging.info("Model loaded !")
    print('*'*10,'Model_name:', model_name)
    print('*'*10,'Model_path:', load_path)

    # load data

    input_path = args.test_dir
    in_files = os.listdir(input_path)
    print('*'*10,'Img number；', len(in_files))
    print('*'*10, 'Img file:', input_path)
    cnt,cnt1,cnt2,cnt3,cnt4,cnt5,cnt6 = 0, 0, 0, 0, 0, 0, 0
    clean = 0
    temp = []
    for i, fn in enumerate(in_files):

        # load data
        file_name = input_path + fn
        temp = Image.open(file_name)
        temp = np.asarray(temp)
        h, w = temp.shape
        if args.direction == 'AtoB':
            img = np.float32(temp[:, 0:h])
            mask_gt = np.float32(temp[:, h:2 * h]/255).squeeze()
        else:
            mask_gt = np.float32(temp[:, 0:h]/255).squeeze()
            img = np.float32(temp[:, h:2 * h])
        # print('mask_gt:', mask_gt[0, 0:10])
        # predict  results

        adv_imgs, adv_masks = generate_adv(net=model,
                           full_img=img,
                           loss=loss,
                           targets=mask_gt,
                           device=device)
        # save results
        # res = np.concatenate((mask,mask_pre),axis=1)
        # print('+++++++',mask_gt.shape)
        # print('+++++++', mask_pre.shape)
        adv_ious = [IoU(mask,mask_gt) for mask in adv_masks]
        # print('+++++++',len(adv_ious))
        # print('+++++++',t)
        logging.info(f'{adv_ious[0]} // {adv_ious[1]} // {adv_ious[2]} // {adv_ious[3]} // {adv_ious[4]} // {adv_ious[5]} // {adv_ious[6]}')

        avg = adv_ious[1]/adv_ious[0] + adv_ious[2]/adv_ious[0] + adv_ious[3]/adv_ious[0] + adv_ious[4]/adv_ious[0] + adv_ious[5]/adv_ious[0] + adv_ious[6]/adv_ious[0]
        cnt = cnt + avg/6

        cnt1 = cnt1 + adv_ious[1]/adv_ious[0]
        cnt2 = cnt2 + adv_ious[2]/adv_ious[0]
        cnt3 = cnt3 + adv_ious[3]/adv_ious[0]
        cnt4 = cnt4 + adv_ious[4]/adv_ious[0]
        cnt5 = cnt5 + adv_ious[5]/adv_ious[0]
        cnt6 = cnt6 + adv_ious[6]/adv_ious[0]
        # cnt1 = cnt1 + adv_ious[1]
        # cnt2 = cnt2 + adv_ious[2]
        # cnt3 = cnt3 + adv_ious[3]
        # cnt4 = cnt4 + adv_ious[4]
        # cnt5 = cnt5 + adv_ious[5]


        clean = clean + adv_ious[0]
        alphas = np.array([0, 1, 2, 4, 8, 16, 32])

        for h in range(len(alphas)):

            adv_img = adv_imgs[h]
            adv_mask = adv_masks[h]
            adv_img = adv_img[:,:,0]
            # print('++++++++++ out', adv_img)

            adv_mask = np.uint8(adv_mask * 255)
            target = np.uint8(mask_gt * 255)
            res = np.concatenate((adv_img, target),axis=1)
            res = Image.fromarray(res)

            save_path = args.attack + '/' + model_name + '/' + args.load_pth + str(alphas[h]) + '_image/'
            os.makedirs(save_path, exist_ok=True)
            res.save(save_path + fn)

            save_path = args.attack + '/' + model_name + '/' + args.load_pth + str(alphas[h])+ '_pred/'
            os.makedirs(save_path, exist_ok=True)
            a, b = os.path.splitext(fn)
            res2 = Image.fromarray(adv_mask)
            target = Image.fromarray(target)
            res2.save(save_path + a + '_pred.png')
            target.save(save_path + a + '_gt.png')

        mean = np.mean([cnt1*100/(i+1), cnt2*100/(i+1), cnt3*100/(i+1), cnt4*100/(i+1),cnt5*100/(i+1),cnt6*100/(i+1)])
        std = np.std([cnt1*100/(i+1), cnt2*100/(i+1), cnt3*100/(i+1), cnt4*100/(i+1), cnt5*100/(i+1),cnt6*100/(i+1)])
        # sum = np.sum([cnt1 / (i + 1), cnt2 / (i + 1), cnt3 / (i + 1), cnt4 / (i + 1), cnt5 / (i + 1)])

    print('------- Clean Acc is ', clean / (i + 1))
    # print('------- Final Robustness mean is', cnt / (i + 1))
    print('------- Final Robustness mean is', mean)
    print('------- Final Robustness std is', std)
    logging.info(f'{cnt1*100/(i+1)} // {cnt2*100/(i+1)} // {cnt3*100/(i+1)} // {cnt4*100/(i+1)} // {cnt5*100/(i+1)}// {cnt6*100/(i+1)}')
    logging.info(f'{load_path}')
    logging.info(f'{save_path}')
