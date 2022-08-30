#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


#------------------------------------------------------------------------------
#   Fundamental losses
#------------------------------------------------------------------------------
def dice_loss(logits, targets, smooth=1.0):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	outputs = F.softmax(logits, dim=1)
	targets = torch.unsqueeze(targets, dim=1)
	targets = torch.zeros_like(logits).scatter_(dim=1, index=targets.type(torch.int64), src=torch.tensor(1.0))

	inter = outputs * targets
	dice = 1 - ((2*inter.sum(dim=(2,3)) + smooth) / (outputs.sum(dim=(2,3))+targets.sum(dim=(2,3)) + smooth))
	return dice.mean()


def dice_loss_with_sigmoid(sigmoid, targets, smooth=1.0):
	"""
	sigmoid: (torch.float32)  shape (N, 1, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1}
	"""
	outputs = torch.squeeze(sigmoid, dim=1)

	inter = outputs * targets
	dice = 1 - ((2*inter.sum(dim=(1,2)) + smooth) / (outputs.sum(dim=(1,2))+targets.sum(dim=(1,2)) + smooth))
	dice = dice.mean()
	return dice


def ce_loss(logits, targets):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	# targets = targets.type(torch.int64)
	targets = targets.type(torch.int64)
	targets = torch.squeeze(targets)
	ce_loss = F.cross_entropy(logits, targets)
	return ce_loss


#------------------------------------------------------------------------------
#   Custom loss for BiSeNet
#------------------------------------------------------------------------------
def custom_bisenet_loss(logits, targets):
	"""
	logits: (torch.float32) (main_out, feat_os16_sup, feat_os32_sup) of shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		main_loss = ce_loss(logits[0], targets)
		os16_loss = ce_loss(logits[1], targets)
		os32_loss = ce_loss(logits[2], targets)
		return main_loss + os16_loss + os32_loss
	else:
		return ce_loss(logits, targets)


#------------------------------------------------------------------------------
#   Custom loss for PSPNet
#------------------------------------------------------------------------------
class PSPNetLoss(nn.Module):
	def __init__(self, alpha=0.4):
		super().__init__()

		"""
		logits: (torch.float32) (main_out, aux_out) of shape (N, C, H, W), (N, C, H/8, W/8)
		targets: (torch.long) shape (N, H, W), value {0,1,...,C-1}
		"""
		self.alpha = alpha
		self.criterion = nn.BCEWithLogitsLoss()

	def forward(self,logits,targets):
		if type(logits) == tuple:
			with torch.no_grad():
				# _targets = torch.unsqueeze(targets, dim=1)
				targets = targets.type(torch.float)
				if len(targets.shape) < 4:
					targets = targets.unsqueeze(1)
				_targets = targets

				aux_targets = F.interpolate(_targets, size=logits[1].shape[-2:], mode='bilinear', align_corners=True)[:,0:1, ...]
			# print('+++++++', aux_targets.size())
			main_loss = self.criterion(logits[0], targets)
			aux_loss = self.criterion(logits[1], aux_targets)
			return main_loss + self.alpha * aux_loss
		else:
			return ce_loss(logits, targets)

#------------------------------------------------------------------------------
#   Custom loss for ICNet
#------------------------------------------------------------------------------
class ICNetLoss(nn.Module):

	def __init__(self, alpha=[0.4, 0.16]):
		super().__init__()
		"""
		logits: (torch.float32)
			[train_mode] (x_124_cls, x_12_cls, x_24_cls) of shape
							(N, C, H/4, W/4), (N, C, H/8, W/8), (N, C, H/16, W/16)

			[valid_mode] x_124_cls of shape (N, C, H, W)

		targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
		
		"""
		self.alpha = alpha
		self.criterion = nn.BCEWithLogitsLoss()

	def forward(self,logits,targets):

		if type(logits)==tuple:
			with torch.no_grad():
				# targets = torch.unsqueeze(targets, dim=0)
				target1 = F.interpolate(targets, size=logits[0].shape[-2:], mode='bilinear', align_corners=True)
				target2 = F.interpolate(targets, size=logits[1].shape[-2:], mode='bilinear', align_corners=True)
				target3 = F.interpolate(targets, size=logits[2].shape[-2:], mode='bilinear', align_corners=True)
			#
			# target1 = target1.type(torch.long)
			# target1 = torch.squeeze(target1).long()
			# target2 = torch.squeeze(target2).long()
			# target3 = torch.squeeze(target3).long()
			# print('++++++++', logits[0].size())
			# print('++++++++', target1.size())
			loss1 = self.criterion(logits[0], target1)
			loss2 = self.criterion(logits[1], target2)
			loss3 = self.criterion(logits[2], target3)
			# print('++++++',loss1)
			return loss1 + float(self.alpha[0])*loss2 + float(self.alpha[1])*loss3

		else:
			return ce_loss(logits, targets)