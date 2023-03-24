""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Full assembly of the parts to form the complete network """



class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		# encoder
		self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
		self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
		self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
		self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
		# decoder
		self.fc1 = nn.Linear(1*1*512, 4*4*256)
		self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
		self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
		self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
		# MLP
		# Position
		self.branch1_fc1 = nn.Linear(512*32*32, 32)
		self.branch1_fc2 = nn.Linear(32, 32)
		self.branch1_fc3 = nn.Linear(32, 32*32*300)
		# Curvature
		self.branch2_fc1 = nn.Linear(512*32*32, 32)
		self.branch2_fc2 = nn.Linear(32, 32)
		self.branch2_fc3 = nn.Linear(32, 32*32*100)
		self.outc = OutConv(512, 3)
		
	def forward(self, x):
		# encoder
		x = F.relu(self.conv1(x)) # (batch_size, 32, 128, 128)
		x = F.relu(self.conv2(x)) # (batch_size, 64, 64, 64)
		x = F.relu(self.conv3(x)) # (batch_size, 128, 32, 32)
		x = F.relu(self.conv4(x)) # (batch_size, 256, 16, 16)
		x = F.relu(self.conv5(x)) # (batch_size, 512, 8, 8)
		x = F.max_pool2d(x, 8) # (batch_size, 512, 1, 1)
		# decoder
		x = x.view(-1, 1*1*512)
		x = F.relu(self.fc1(x))
		x = x.view(-1, 256, 4, 4)
		x = F.relu(self.conv6(x)) # (batch_size, 512, 4, 4)
		x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False) # (batch_size, 512, 8, 8)
		x = F.relu(self.conv7(x)) # (batch_size, 512, 8, 8)
		x = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners = False) # (batch_size, 512, 16, 16)
		x = F.relu(self.conv8(x)) # (batch_size, 512, 16, 16)
		x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False) # (batch_size, 512, 32, 32)
		x = self.outc(x)
		x = x.view(x.size(0),-1,3)
		# x = x.view(-1, 512*32*32)
		# MLP
		return x


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Change here to adapt to your data
	# n_channels=3 for RGB images
	# n_classes is the number of probabilities you want to get per pixel
	net = UNet()
	net.to(device=device)
	input_tensor = torch.rand(1, 3, 256, 256, device=device)
	output = net(input_tensor)
	print(output.shape)