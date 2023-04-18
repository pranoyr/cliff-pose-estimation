import numpy as np
import os, glob, cv2, sys
from torch.utils.data import Dataset


import torch
from .models.pose_2D import KeypointRCNN
from .common.renderer_pyrd import Renderer
from .common.imutils import process_image
from .common.utils import estimate_focal_length
from torchvision.transforms import Normalize
from .utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from .utils.cliff_utils import convert_to_angle_axis, convert_to_rotmat

import pickle as pk
from PIL import Image



def augm_params():
	"""Get augmentation parameters."""
	flip = 0            # flipping
	pn = np.ones(3)  # per channel pixel-noise
	rot = 0            # rotation
	sc = 1            # scaling
	
	noise_factor = 0.4 
	rot_factor = 30
	scale_factor = 0.25

   
	# We flip with probability 1/2
	if np.random.uniform() <= 0.5:
		flip = 1
	
	# Each channel is multiplied with a number 
	# in the area [1-opt.noiseFactor,1+opt.noiseFactor]
	pn = np.random.uniform(1-noise_factor, 1+noise_factor, 3)
	
	# The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
	rot = min(2*rot_factor,
			max(-2*rot_factor, np.random.randn()*rot_factor))
	
	# The scale is multiplied with a number
	# in the area [1-scaleFactor,1+scaleFactor]
	sc = min(1+scale_factor,
			max(1-scale_factor, np.random.randn()*scale_factor+1))
	# but it is zero with probability 3/5
	if np.random.uniform() <= 0.6:
		rot = 0
	
	return flip, pn, rot, sc



def pose_processing(pose, r, f):
	"""Process SMPL theta parameters  and apply all augmentation transforms."""
	# rotation or the pose parameters
	pose[:3] = rot_aa(pose[:3], r)
	# flip the pose parameters
	if f:
		pose = flip_pose(pose)
	# (72),float
	pose = pose.astype('float32')
	return pose



class CustomDataset(Dataset):
	def __init__(self, cfg, transform = None, target_transform=None, image_set='train', img_format='jpg'):
		self.cfg = cfg
		self.transform = transform
		self.target_transform = target_transform
		if image_set == 'train':
			root_dir = cfg.DATA.TRAIN_PATH
		else:
			root_dir = cfg.DATA.VAL_PATH
		self.root_dir = os.path.join(root_dir, "images")
		self.image_set = image_set
		self.aug = True
		self.img_format = img_format
		self.image_list = self.get_data(self.root_dir)
		self.keypoint_detector = KeypointRCNN()
		self.setting_cache()
	
	def get_data(self, data_path):
		data = []
		for img_path in glob.glob(data_path + os.sep + '*'):
			data.append(img_path)
		return data

	def _read_image_ids(self, image_sets_file):
		ids = []
		with open(image_sets_file) as f:
			for line in f:
				ids.append(line.rstrip())
		return ids
	
	def check_cache(self, img_path):
		if img_path in self.map_dict:
			return True
		else:
			return False
		
	def setting_cache(self):
		# loop over cache and setting map_dict
		self.map_dict = {}
		for file in glob.glob(self.cfg.CACHE + os.sep + '*'):
			img_path = os.path.join(self.root_dir, file.split("/")[-1].split(".")[0]+".jpg")
			self.map_dict[img_path] = file
		
	def save_to_cache(self, img_path, kpts):
		write_path = os.path.join(self.cfg.CACHE , img_path.split("/")[-1].split(".")[0]+".pkl")
		self.map_dict[img_path] = write_path
		if not os.path.exists(write_path):
			pk.dump(kpts, open(write_path, "wb"))
	
	def load_from_cache(self, img_path):
		file = pk.load(open(self.map_dict[img_path], "rb"))
		return file
		
	
	def __getitem__(self, index):  
		flip, pn, rot, sc = augm_params()
		flip=0
		img_path = self.image_list[index]
		params_path = img_path.replace("images", "params").replace(self.img_format, "pkl")

		img = Image.open(img_path)
		img_rgb = img.convert('RGB')
		img_rgb = np.array(img_rgb)

		# get the image size
		img_h, img_w, _ = img_rgb.shape
		
		# if params file exists, load it
		if os.path.exists(params_path):
			with open(params_path, 'rb') as f:
				data = pk.load(f)
				pose_params = data['body_pose']
				beta_params = data['beta']
				is_params = 1
		else:
			pose_params = torch.zeros((23, 3, 3))
			beta_params = torch.zeros((10,))
			is_params = 0
		

		if self.check_cache(img_path):
			keypoint_results = self.load_from_cache(img_path)
		else:
			keypoint_results = self.keypoint_detector.detect_pose(img_rgb)
			self.save_to_cache(img_path, keypoint_results)

		
		target_landmarks = keypoint_results["normalised_keypoints"]
		focal_length = estimate_focal_length(img_h, img_w)
		bbox = keypoint_results["bbox"]
		norm_img, center, scale, _, _, _ = process_image(self.cfg, img_rgb, bbox)


		if self.image_set == 'train' and is_params == 1:
			scaled_keypoints = keypoint_results["scaled_keypoints"]
			matrix = cv2.getRotationMatrix2D(((img_w - 1) * 0.5, (img_h - 1) * 0.5), rot, 1.0)
			keypoints = [torch.from_numpy(cv2.transform(np.array([[[keypoint[0], keypoint[1]]]]), matrix).squeeze()) for keypoint in scaled_keypoints]
			keypoints = torch.stack(keypoints, dim=0)

			target_landmarks[:,0] = keypoints[:,0] / img_w
			target_landmarks[:,1] = keypoints[:,1] / img_h

			norm_img, center, scale, _, _, crop_img = process_image(self.cfg, img_rgb, bbox, pn=pn, rot=rot, flip=flip, train=True)

			pose_params = convert_to_angle_axis(torch.from_numpy(pose_params).unsqueeze(0), 23)
			pose_params = torch.from_numpy(pose_processing(pose_params.numpy(), rot, flip)).float()
			pose_params = convert_to_rotmat(pose_params, 23)
			beta_params = torch.from_numpy(beta_params)



		data = {}
		data["norm_img"] = norm_img
		data["center"] = center
		data["scale"] = scale
		data["focal_length"] = focal_length
		data["img_h"] = img_h
		data["img_w"] = img_w
		data["pose_params"] = pose_params
		data["beta_params"] = beta_params
		data["target_landmarks"] = target_landmarks
		data['is_params'] = is_params

		return data

	def __len__(self):
		return len(self.image_list)
	


if __name__ == "__main__":

	train_list = "/home/pranoy/datasets/human3.6M/train.txt"
	root_dir = "/home/pranoy/datasets/human3.6M"
	train_dataset = CustomDataset(root_dir, train_list, image_set="train")
	train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=32, shuffle=True,
			num_workers=1, pin_memory=False)

	for i, data in enumerate(train_loader):
		print(i)
		

