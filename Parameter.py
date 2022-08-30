import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='FCNs',
                        choices=['FCNs','UNet','SegNet','simple_unet','UNet_3','DeepLab',
                                 'PSPNet','ICNet'],help='which network to train')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--lr', metavar='LR', type=float, nargs='?', default=1e-2,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--save_pth', type=str, default='trained_on_mito/',
                        help='save weights to...')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-g','--gpu', type=int, default=0,
                        help='which GPU to be used. Sush as -gpu 0')
    parser.add_argument('--train_dir', default='data/mito/train/', type=str,
                        help='filenames of train images', required=False)
    parser.add_argument('--val_dir', default='data/mito/val/', type=str,
                        help='filenames of val images', required=False)
    parser.add_argument('--direction', type=str, default = 'AtoB', help='AtoB or BtoA')
    parser.add_argument('--norm', type=str, default='std', help='which normalization...')
    parser.add_argument('--epsilon', '-eps', type=float, default=8,
                        help='maximum perturbation of adversaries (8/255)')
    parser.add_argument('--alpha', '-a', type=float, default=2,
                        help='movement multiplier per iteration when generating adversarial examples (2/255)')
    parser.add_argument('--k', '-k', type=int, default=10,
                        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf',
                        help='the type of the perturbation (linf or l2)')

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))