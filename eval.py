import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import *

def miou(logits, targets, eps=1e-6):
    """
    logits: (torch.float32)  shape (N, C, H, W)
    targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
    """

    outputs = torch.argmax(logits, dim=1, keepdim=True).type(torch.int64)

    targets = torch.unsqueeze(targets, dim=1).type(torch.int64)
    # print('+++++++++', targets.size())
    outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.tensor(1.0)).type(torch.int8)
    targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.int8)

    inter = (outputs & targets).type(torch.float32).sum(dim=(2,3))
    union = (outputs | targets).type(torch.float32).sum(dim=(2,3))

    iou = inter / (union + eps)

    return iou.mean()

def custom_pspnet_miou(logits, targets):
    """
    logits: (torch.float32) (main_out, aux_out) of shape (N, C, H, W), (N, C, H/8, W/8)
    targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
    """

    if type(logits)==tuple:
        return miou(logits[0], targets)
    else:
        return miou(logits, targets)

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    # print('mask_type:', mask_type)
    # mask_type = torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.output_c > 1:
                pred = F.softmax(mask_pred, dim=1)
                # probs = torch.argmax(probs, dim=1, keepdim=True)
                pred = pred[:, 1:2, :, :].float()

            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()

            tot += iou_coeff(pred, true_masks).item()
            pbar.update()

    return tot / n_val
