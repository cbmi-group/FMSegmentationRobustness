import argparse
import logging
import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.FCN import FCNs,VGGNet
from models.UNet_5 import UNet
from models.UNet_3 import UNet_3
from models.segnet import SegNet
from models.simple_unet import simple_unet
from models.DeepLab.deeplab_v3 import DeepLab
from models.PSPNet.PSPNet import PSPNet
from models.ICNet.ICNet import ICNet

from dataset import BasicDataset
from PIL import Image

# from feature_function import FeatureVisualization

# logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def predict_img(net,
                full_img,
                device,
                norm,
                out_threshold):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, norm))
    # print(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():

        output = net(img)

        if net.output_c > 1:
            probs = F.softmax(output, dim=1)
            # probs = torch.argmax(probs, dim=1, keepdim=True)
            probs = probs[:,1:2,:,:]
        else:
            probs = torch.sigmoid(output)

        # probs = probs.squeeze()
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]

        )
        # probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='FCNs', choices=['FCNs','UNet','UNet_3', 'simple_unet',
                                                                      'SegNet','DeepLab','PSPNet','ICNet'],
                        help='Which network to train')
    parser.add_argument('--test_dir', default='data/mito/test/',type=str, help='directory of test images')
    parser.add_argument('--load_pth', type=str,default='trained_on_mito/',
                        help='model weights directory')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--mask-threshold', type=float, help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to be used. Such as -gpu 0')
    parser.add_argument('--direction', type=str, default='AtoB',
                        help='AtoB or BtoA')
    parser.add_argument('--best', type=bool, default=False,help='Whether to load the best model')
    parser.add_argument('--norm', type=str, default='std', choices=['std','255'], help='which normalization...')

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []
    print('=======',len(args.input))
    print('=======', len(args.output))
    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


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
    # os.environ['CUDA_VISIBLE_DEVICE'] = '0,1'
    torch.cuda.set_device(args.gpu)
    model_name = args.model

    # load model networks
    if model_name == 'FCNs':
        vgg_model = VGGNet(requires_grad=True)
        model = eval(model_name)(pretrained_net=vgg_model)
    else:
        model = eval(model_name)()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    logging.info("Model loaded !")
    print('Model_name:', model_name)

    if args.best:
        load_path = 'checkpoints/' + model_name + '/' + args.load_pth + 'model_best.pth'
    else:
        load_path = 'checkpoints/' + model_name + '/' + args.load_pth + 'CP_epoch' + str(args.epoch) + '.pth'
    model.load_state_dict(torch.load(load_path, map_location=device), strict=False)

    # print('Model_path:', load_path)
    # input path

    input_path = args.test_dir
    in_files = os.listdir(input_path)
    sub_path = args.test_dir.split('/')[-2]
    # print('+'*10,sub_path)
    # save path
    save_path = 'results/' + model_name + '/' + args.load_pth + sub_path + '/'
    os.makedirs(save_path, exist_ok=True)
    # print('saving to:', save_path)
    iou = 0
    for i, fn in enumerate(in_files):
        # if i % 20 ==0:
        #     print(i)
        file_name = input_path + fn
        # logging.info("\nPredicting image {} ...".format(fn))

        # load data
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
        mask_pre = predict_img(net=model,
                           full_img=img,
                           norm=args.norm,
                           out_threshold=args.mask_threshold,
                           device=device)
        # save results

        iou = iou + IoU(mask_pre, mask_gt)

        mask_pre = np.uint8(mask_pre * 255)
        mask_gt = np.uint8(mask_gt * 255)
        res1 = Image.fromarray(mask_gt)
        res1.save(save_path + fn[0:-4] + '_gt' + fn[-4:])
        res2 = Image.fromarray(mask_pre)
        res2.save(save_path + fn[0:-4] + '_pred' + fn[-4:])
    iou = iou/(i + 1)
    print('*'*60)
    logging.info(f'Test data:, {input_path}')
    logging.info(f'Mean IoU: {iou}')
    logging.info(f'Load_path:, {load_path}')