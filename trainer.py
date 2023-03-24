import torch
import numpy as np
import wandb
import os
import os
import sys
import torch
from utils.cliff_utils import *
import cv2
import torch.nn as nn
from torch.nn import functional as F
from common.imutils import process_image
from common.utils import estimate_focal_length
from common.renderer_pyrd import Renderer
from losses import *

import random

from utils.meter import *

import albumentations as A

import os
import torch

# add path for demo utils functions
import sys
import os
import numpy as np
import smplx
import math

from dataset import CustomDataset
import wandb
from models.pose_2D import KeypointRCNN


class Trainer():
	""" Trainer class for training and validation"""

	def __init__(self, cfg, model, device):
		super().__init__()
		self.cfg = cfg
		self.device = device
		self.model = model

		self.start_epoch = 0

		self.prepare_data()
		self.get_training_config()
		self.resume_training()
		self.smpl_model = smplx.create(
			cfg.SMPL.SMPL_MODEL_DIR, "smpl").to(device)

		self.th = math.inf
		self.keypoint_model  = KeypointRCNN()

	def init_meter(self):
		"""Init meter for training and validation"""

		self.losses = AverageMeter('Loss', ':.4f')

		self.progress = ProgressMeter(
			len(self.train_loader) * self.cfg.TRAIN.EPOCHS,
			[self.losses])

	def prepare_data(self):
		"""Prepare data for training and validation"""

		train_dataset = CustomDataset(self.cfg, image_set="train")
		val_dataset = CustomDataset(self.cfg, image_set="val")
		self.train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=self.cfg.DATA.NUM_WORKERS)
		self.val_loader = torch.utils.data.DataLoader(
			val_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=False, num_workers=self.cfg.DATA.NUM_WORKERS)
		print("train dataset size: ", len(train_dataset))
		print("val dataset size: ", len(val_dataset))
		print("Total Iterations: ", len(
			self.train_loader) * self.cfg.TRAIN.EPOCHS)

		# init meter
		self.init_meter()

	def get_training_config(self):
		"""Get training config"""
		
		if self.cfg.TRAIN.OPTIMIZER.NAME == 'adam':
			self.optimizer = torch.optim.Adam(self.model.parameters(
			), lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
		elif self.cfg.TRAIN.OPTIMIZER.NAME == 'adamr':
			self.optimizer = torch.optim.AdamR(self.model.parameters(
			), lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
		elif self.cfg.TRAIN.OPTIMIZER.NAME == 'sgd':
			self.optimizer = torch.optim.SGD(self.model.parameters(
			), lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
		elif self.cfg.TRAIN.OPTIMIZER.NAME == 'adamw':
			self.optimizer = torch.optim.AdamW(self.model.parameters(
			), lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

		if self.cfg.TRAIN.LR_SCHEDULER.NAME == 'multistep':
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
				self.optimizer, milestones=self.cfg.TRAIN.LR_SCHEDULER.MULTISTEPS, gamma=self.cfg.TRAIN.LR_SCHEDULER.GAMMA)
		# linear scheduler
		elif self.cfg.TRAIN.LR_SCHEDULER.NAME == 'linear':
			self.scheduler = torch.optim.lr_scheduler.LambdaLR(
				self.optimizer, lr_lambda=lambda epoch: 1 - epoch / self.cfg.TRAIN.EPOCHS)

	def resume_training(self):
		"""Resume training from checkpoint"""

		if self.cfg.MODEL.RESUME:
			checkpoint = torch.load(self.cfg.MODEL.RESUME)
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.scheduler.load_state_dict(checkpoint['scheduler'])
			self.start_epoch = checkpoint['epoch'] + 1
			print(
				f"==> Loaded checkpoint '{self.cfg.MODEL.RESUME}' (epoch {self.start_epoch})")
		elif self.cfg.MODEL.PRETRAINED:
			state_dict  = torch.load(self.cfg.MODEL.PRETRAINED)['model']
			self.model.load_state_dict(state_dict, strict=False)
			print( f"==> Loaded pretrained weights for backbone '{self.cfg.MODEL.PRETRAINED}'")

	def visualise(self):
		"""Visualise validation results"""

		# load image
		files = os.listdir(self.cfg.DATA.VAL_PATH + "/images")
		random_file = random.choice(files)
		img_bgr = cv2.imread(os.path.join(self.cfg.DATA.VAL_PATH, "images", random_file))
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

		kpt_results = self.keypoint_model.detect_pose(img_rgb)  # shape (18, 2)
		bbox = kpt_results["bbox"].cuda().float()

		img_h, img_w, _ = img_rgb.shape
		focal_length = estimate_focal_length(img_h, img_w)

		norm_img, center, scale, crop_ul, crop_br, _ = process_image(
			img_rgb, bbox)

		center = center.unsqueeze(0).to(self.device)
		scale = scale.unsqueeze(0)
		focal_length = torch.tensor([focal_length]).to(self.device)

		cx, cy, b = center[:, 0], center[:, 1], scale * 200

		bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
		# The constants below are used for normalization, and calculated from H36M data.
		# It should be fine if you use the plain Equation (5) in the paper.
		bbox_info[:, :2] = bbox_info[:, :2] / \
			focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
		bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 *
						   focal_length) / (0.06 * focal_length)  # [-1, 1]

		norm_img = torch.from_numpy(norm_img).unsqueeze(0)
		norm_img = norm_img.to(self.device)

		with torch.no_grad():
			pred_rotmat, pred_betas, pred_cam_crop = self.model(
				norm_img, bbox_info)

		img_shape = torch.tensor([[img_h, img_w]]).float().to(self.device)

		pred_cam_full = cam_crop2full(
			pred_cam_crop, center, scale, img_shape, focal_length)

		pred_output = self.smpl_model(betas=pred_betas,
									  body_pose=pred_rotmat[:, 1:],
									  global_orient=pred_rotmat[:, [0]],
									  pose2rot=False,
									  transl=pred_cam_full)

		vertices = pred_output.vertices

		img_h, img_w, _ = img_bgr.shape
		img_h = torch.tensor([img_h]).float().to(self.device)
		img_w = torch.tensor([img_w]).float().to(self.device)

		focal_length = estimate_focal_length(img_h, img_w)

		renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
								 faces=self.smpl_model.faces,
								 same_mesh_color=("video" == "video"))

		front_view = renderer.render_front_view(vertices.cpu(), img_bgr)
		front_view = cv2.resize(front_view, (480, 640))
		output_file_path = self.cfg.OUTPUT_DIR + f"/result_{self.cfg.EXP_NAME}.jpg"
		print(output_file_path)
		cv2.imwrite(output_file_path, front_view)
		del renderer

	def validate(self):
		"""Validate the model"""

		losses = 0
		for i, (batch) in enumerate(self.val_loader):
			norm_img = batch["norm_img"].to(self.device).float()
			center = batch["center"].to(self.device).float()
			scale = batch["scale"].to(self.device).float()
			img_h = batch["img_h"].to(self.device).float()
			img_w = batch["img_w"].to(self.device).float()
			focal_length = batch["focal_length"].to(self.device).float()
			is_params = batch["is_params"].to(self.device).float()
			target_landmarks = batch["target_landmarks"].to(self.device).float()
			pose_params = batch["pose_params"].to(self.device).float()
			beta_params = batch["beta_params"].to(self.device).float()

			cx, cy, b = center[:, 0], center[:, 1], scale * 200
			bbox_info = torch.stack(
				[cx - img_w / 2., cy - img_h / 2., b], dim=-1)
			# The constants below are used for normalization, and calculated from H36M data.
			# It should be fine if you use the plain Equation (5) in the paper.
			bbox_info[:, :2] = bbox_info[:, :2] / \
				focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
			bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 *
							   focal_length) / (0.06 * focal_length)  # [-1, 1]

			pred_rotmat, pred_betas, pred_cam_crop = self.model(
				norm_img, bbox_info)

			# convert the camera parameters from the crop camera to the full camera
			img_shape = torch.stack((img_h, img_w), dim=-1)
			pred_cam_full = cam_crop2full(
				pred_cam_crop, center, scale, img_shape, focal_length)

			pred_output = self.smpl_model(betas=pred_betas,
										  body_pose=pred_rotmat[:, 1:],
										  global_orient=pred_rotmat[:, [0]],
										  pose2rot=False,
										  transl=pred_cam_full)

			pred_joints = pred_output.joints

			projected_keypoints_2d = perspective_projection(pred_joints,
															rotation=torch.eye(3, device="cuda:0").unsqueeze(
																0).expand(1, -1, -1),
															translation=pred_cam_full,
															focal_length=focal_length,
															camera_center=torch.div(img_shape.flip(dims=[1]), 2, rounding_mode='floor'))

			smplx_left_leg_indices = torch.tensor([2, 5, 8])
			smplx_right_leg_indices = torch.tensor([1, 4, 7])
			smplx_left_arm_indices = torch.tensor([17, 19, 21])
			smplx_right_arm_indices = torch.tensor([16, 18, 20])
			nose_neck_indices = torch.tensor([15])

			all_smplx_indices = torch.cat((smplx_left_leg_indices, smplx_right_leg_indices,
										  smplx_left_arm_indices, smplx_right_arm_indices, nose_neck_indices), dim=0)

			projected_keypoints_2d = projected_keypoints_2d[:, all_smplx_indices, :] / img_shape.unsqueeze(1)

			kpt_right_leg_indices = torch.tensor([11, 13, 15])
			kpt_left_leg_indices = torch.tensor([12, 14, 16])
			kpt_right_arm_indices = torch.tensor([5, 7, 9])
			kpt_left_arm_indices = torch.tensor([6, 8, 10])
			kpt_nose_neck_indices = torch.tensor([0])

			all_kpt_indices = torch.cat((kpt_left_leg_indices, kpt_right_leg_indices,
											  kpt_left_arm_indices, kpt_right_arm_indices, kpt_nose_neck_indices), dim=0)
			target_landmarks = target_landmarks[:,all_kpt_indices, :]

			keypoint_loss = compute_keypoint_loss(projected_keypoints_2d, target_landmarks)
			beta_loss = compute_beta_loss(pred_betas, beta_params, is_params)
			pose_loss = compute_pose_loss(pred_rotmat, pose_params, is_params)

			keypoint_loss = keypoint_loss * self.cfg.MODEL.KEY_LOSS_WEIGHT
			beta_loss = beta_loss * self.cfg.MODEL.BETA_LOSS_WEIGHT
			pose_loss = pose_loss * self.cfg.MODEL.POSE_LOSS_WEIGHT

			loss = keypoint_loss + beta_loss + pose_loss + \
				((torch.exp(-pred_cam_crop[:, 0]*10)) ** 2).mean()

			loss *= 60

			losses += loss.item()

		return losses / len(self.val_loader)

	def save_checkpoint(self, epoch, filename, is_best=False):
		"""Save checkpoint"""

		if is_best:
			checkpoint = {
				'epoch': epoch,
				'state_dict': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'scheduler': self.scheduler.state_dict()
			}
			torch.save(checkpoint, filename)

	def fit(self):
		"""Train the model"""

		self.model.train()
		for epoch in range(self.start_epoch, self.cfg.TRAIN.EPOCHS):
			for idx, batch in enumerate(self.train_loader):
				step = epoch * len(self.train_loader) + idx
				norm_img = batch["norm_img"].to(self.device).float()
				center = batch["center"].to(self.device).float()
				scale = batch["scale"].to(self.device).float()
				img_h = batch["img_h"].to(self.device).float()
				img_w = batch["img_w"].to(self.device).float()
				focal_length = batch["focal_length"].to(self.device).float()
				is_params = batch["is_params"].to(self.device).float()
				target_landmarks = batch["target_landmarks"].to(self.device).float()
				pose_params = batch["pose_params"].to(self.device).float()
				beta_params = batch["beta_params"].to(self.device).float()

				cx, cy, b = center[:, 0], center[:, 1], scale * 200
				bbox_info = torch.stack(
					[cx - img_w / 2., cy - img_h / 2., b], dim=-1)
				# The constants below are used for normalization, and calculated from H36M data.
				# It should be fine if you use the plain Equation (5) in the paper.
				bbox_info[:, :2] = bbox_info[:, :2] / \
					focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
				bbox_info[:, 2] = (
					bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

				# run model
				pred_rotmat, pred_betas, pred_cam_crop = self.model(
					norm_img, bbox_info)

				# convert the camera parameters from the crop camera to the full camera
				img_shape = torch.stack((img_h, img_w), dim=-1)
				pred_cam_full = cam_crop2full(
					pred_cam_crop, center, scale, img_shape, focal_length)

				# run SMPL model
				pred_output = self.smpl_model(betas=pred_betas,
											  body_pose=pred_rotmat[:, 1:],
											  global_orient=pred_rotmat[:, [
												  0]],
											  pose2rot=False,
											  transl=pred_cam_full)
				pred_joints = pred_output.joints
				projected_keypoints_2d = perspective_projection(pred_joints,
																rotation=torch.eye(3, device="cuda").unsqueeze(
																	0).expand(1, -1, -1),
																translation=pred_cam_full,
																focal_length=focal_length,
																camera_center=torch.div(img_shape.flip(dims=[1]), 2, rounding_mode='floor'))

				smplx_left_leg_indices = torch.tensor([2, 5, 8])
				smplx_right_leg_indices = torch.tensor([1, 4, 7])
				smplx_left_arm_indices = torch.tensor([17, 19, 21])
				smplx_right_arm_indices = torch.tensor([16, 18, 20])
				nose_neck_indices = torch.tensor([15])

				img_shape_1 = torch.stack(
					(img_w, img_h), dim=-1).unsqueeze(1)
				all_smplx_indices = torch.cat((smplx_left_leg_indices, smplx_right_leg_indices,
											  smplx_left_arm_indices, smplx_right_arm_indices, nose_neck_indices), dim=0)
				projected_keypoints_2d = projected_keypoints_2d[:,
																all_smplx_indices, :] / img_shape_1

				kpt_right_leg_indices = torch.tensor([11, 13, 15])
				kpt_left_leg_indices = torch.tensor([12, 14, 16])
				kpt_right_arm_indices = torch.tensor([5, 7, 9])
				kpt_left_arm_indices = torch.tensor([6, 8, 10])
				kpt_nose_neck_indices = torch.tensor([0])

				all_kpt_indices = torch.cat((kpt_left_leg_indices, kpt_right_leg_indices,
												  kpt_left_arm_indices, kpt_right_arm_indices, kpt_nose_neck_indices), dim=0)
				target_landmarks = target_landmarks[:,all_kpt_indices, :]

				keypoint_loss = compute_keypoint_loss(projected_keypoints_2d, target_landmarks)
				beta_loss = compute_beta_loss(pred_betas, beta_params, is_params)
				pose_loss = compute_pose_loss(pred_rotmat, pose_params, is_params)

				keypoint_loss = keypoint_loss * self.cfg.MODEL.KEY_LOSS_WEIGHT
				beta_loss = beta_loss * self.cfg.MODEL.BETA_LOSS_WEIGHT
				pose_loss = pose_loss * self.cfg.MODEL.POSE_LOSS_WEIGHT

				loss = keypoint_loss + beta_loss + pose_loss + \
					((torch.exp(-pred_cam_crop[:, 0]*10)) ** 2).mean()

				loss *= 60
				self.losses.update(loss.item(), norm_img.size(0))

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				# LOGGING
				if step % self.cfg.PRINT_FREQ == 0:
					self.progress.display(step)

				if step % self.cfg.TRAIN_FREQ == 0:
					wandb.log({'train_loss': self.losses.avg}, step=step)
		
				if step % self.cfg.VALID_FREQ == 0:
					self.model.eval()
					val_loss = self.validate()
					self.visualise()
					self.model.train()

					lr = self.optimizer.param_groups[0]['lr']
					wandb.log({'train_loss': self.losses.avg}, step=step)
					wandb.log({'val_loss': val_loss}, step=step)
					wandb.log({'lr': lr}, step=step)

					is_best = val_loss < self.th
					self.th = min(val_loss, self.th)
					checkpoint_path = self.cfg.CKPT_DIR + f"/best_weights_{self.cfg.EXP_NAME}.pth"
					self.save_checkpoint(
						epoch,  checkpoint_path, is_best=is_best)

				if step % self.cfg.SAVE_FREQ == 0:
					# save model
					# checkpoint_iter10_trial1.pth
					checkpoint_name = self.cfg.CKPT_DIR + f"/checkpoint_iter{step}_{self.cfg.EXP_NAME}.pth"
					self.save_checkpoint(epoch, checkpoint_name,  is_best=True)
					print("model saved to {}".format(checkpoint_name))
					print()

			print("Epoch " + str(epoch) + " completed")
			print()

		# Save the final model
		# final_weights_trial1.pth
		checkpoint_name = self.cfg.CKPT_DIR + f"/final_weights_{self.cfg.EXP_NAME}.pth"
		self.save_checkpoint(epoch, checkpoint_name, is_best=True)
		