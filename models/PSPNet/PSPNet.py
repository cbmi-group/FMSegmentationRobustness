#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .base_model import BaseModel
from .ResNet import *


#------------------------------------------------------------------------------
#  Convolutional block
#------------------------------------------------------------------------------
class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, input):
		x = self.conv(input)
		x = self.bn(x)
		x = F.relu(x, inplace=True)
		return x


#------------------------------------------------------------------------------
#  Pyramid Pooling Module
#------------------------------------------------------------------------------
class PyramidPoolingModule(nn.Module):
	def __init__(self, in_channels, pyramids=[1,2,3,6]):
		super(PyramidPoolingModule, self).__init__()
		self.pyramids = pyramids
		out_channels = in_channels // len(pyramids)

		self.convs = nn.ModuleList()
		for _ in pyramids:
			conv = ConvBlock(in_channels, out_channels, kernel_size=1, bias=False)
			self.convs.append(conv)

	def forward(self, input):
		feat = [input]
		height, width = input.shape[2:]
		for i, bin_size in enumerate(self.pyramids):
			x = F.adaptive_avg_pool2d(input, output_size=bin_size)
			x = self.convs[i](x)
			x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
			feat.append(x)
		x = torch.cat(feat, dim=1)  # 把list中的tensor按通道拼接起来
		return x


#------------------------------------------------------------------------------
#  PSPNet
#------------------------------------------------------------------------------
class PSPNet(BaseModel):
	dropout = 0.1
	pyramids = [1, 2, 3, 6]
	backbone_os = 8

	def __init__(self, backbone='resnet18', num_classes=1, pretrained_backbone=None):
		super(PSPNet, self).__init__()
		if 'resnet' in backbone:
			if backbone=='resnet18':
				n_layers = 18
				stage5_channels = 512
			elif backbone=='resnet34':
				n_layers = 34
				stage5_channels = 512
			elif backbone=='resnet50':
				n_layers = 50
				stage5_channels = 2048
			elif backbone=='resnet101':
				n_layers = 101
				stage5_channels = 2048
			else:
				raise NotImplementedError
			self.features = []
			self.output_c = num_classes
			self.run_backbone = self._run_backbone_resnet
			self.backbone = get_resnet(n_layers, output_stride=self.backbone_os, num_classes=None)
			self.pyramid = PyramidPoolingModule(in_channels=stage5_channels, pyramids=self.pyramids)

			# self.main_output = nn.Sequential(OrderedDict([
			# 	("conv1", ConvBlock(2*stage5_channels, stage5_channels//4, kernel_size=3, padding=1, bias=False)),
			# 	("dropout", nn.Dropout2d(p=self.dropout)),
			# 	("conv2", nn.Conv2d(stage5_channels//4, num_classes, kernel_size=1, bias=False)),
			# ]))
			self.main_output = nn.Sequential(OrderedDict([
				("conv1", ConvBlock(1072, 128, kernel_size=3, padding=1, bias=False)),
				("dropout", nn.Dropout2d(p=self.dropout)),
				("conv2", nn.Conv2d(128, num_classes, kernel_size=1, bias=False)),
			]))
			self.aux_output = nn.Sequential(OrderedDict([
				("conv1", ConvBlock(stage5_channels//2, stage5_channels//8, kernel_size=3, padding=1, bias=False)),
				("dropout", nn.Dropout2d(p=self.dropout)),
				("conv2", nn.Conv2d(stage5_channels//8, num_classes, kernel_size=1, bias=False)),
			]))
		else:
			raise NotImplementedError
		# print('++++++ num_classes', num_classes)

		self.features.append(self.run_backbone)
		self.features.append(self.pyramid)
		self.features.append(self.main_output)
		self.init_weights()
		print('++++++ PSPNet Version is: with_link_128')

		if pretrained_backbone is not None:
			print('++++++++ load pretrained models:', pretrained_backbone)
			self.backbone.load_pretrained_model(pretrained_backbone)

		self.conv1 = nn.Conv2d(64, 48, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(48)
		self.relu = nn.ReLU()

	def forward(self, input):
		inp_shape = input.shape[2:]
		# print('+'*5,input.shape)
		if self.training:
			feat_stage5, feat_stage4, low_level_feat = self.run_backbone(input)  # feat_stage5 (512, 32, 32), low_level_feat (64, 128, 128)
			# print('++++++ low_feat ',low_level_feat.shape)
			feat_pyramid = self.pyramid(feat_stage5)  # feat_pyramid.shape (1024,32,32)
			# main_out = self.main_output(feat_pyramid)
			main_out = self.decoder(feat_pyramid, low_level_feat)  # main_out.shape (num_classes, 128, 128)
			# print(main_out.shape)
			main_out = F.interpolate(main_out, size=inp_shape, mode='bilinear', align_corners=True)  # (num_classes,256,256)
			aux_out = self.aux_output(feat_stage4)
			# print('++++++ main train',main_out.size())
			# print('++++++ aux', aux_out.size())
			return main_out, aux_out

		else:
			feat_stage5, low_level_feat = self.run_backbone(input)
			feat_pyramid = self.pyramid(feat_stage5)
			main_out = self.decoder(feat_pyramid, low_level_feat)
			# main_out = self.main_output(feat_pyramid)
			# print('++++++ main out', main_out.size())
			main_out = F.interpolate(main_out, size=inp_shape, mode='bilinear', align_corners=True)
			# print('++++++ main test', main_out.size())
			return main_out

	def decoder(self, x,low_level_feat):

		low_level_feat = self.conv1(low_level_feat)
		low_level_feat = self.bn1(low_level_feat)
		low_level_feat = self.relu(low_level_feat)  # (48,128,128)
		# print('+++++++ x ',x.shape)
		# print('+++++++ low-feat ', low_level_feat.shape)

		x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)  # (1032,128,128)
		x = torch.cat((x, low_level_feat), dim=1)   # (1024+48,128,128)
		# print('++++++ x ,', x.shape)
		x = self.main_output(x)

		return x

	def _run_backbone_resnet(self, input):
		# Stage1
		# print('*' * 5, input.shape)    # input.shape = (256,256)
		x1 = self.backbone.conv1(input) # (128,128)

		x1 = self.backbone.bn1(x1)
		x1 = self.backbone.relu(x1)

		# Stage2
		x2 = self.backbone.maxpool(x1)  # (64, 64)
		x2 = self.backbone.layer1(x2)   # (64, 64)
		# print('++++++ x2 ,', x2.shape)
		# Stage3
		x3 = self.backbone.layer2(x2)  # (32, 32)

		# Stage4
		x4 = self.backbone.layer3(x3)   # (32, 32)

		# Stage5
		x5 = self.backbone.layer4(x4)   # (32, 32)

		# Output
		if self.training:
			return x5, x4, x1
		else:
			return x5, x1