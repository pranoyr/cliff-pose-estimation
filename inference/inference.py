# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import os
import math
from random import random
import time

from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))


# from render import *
# from pytorch3d.structures import Meshes
import pickle
import os.path as osp
import cv2
import torch

from lib import CLIFF
from lib.models.pose_2D import KeypointRCNN




import argparse
from lib.utils import *
import numpy as np
from tqdm import tqdm
import smplx

import torchgeometry as tgm


from lib.common.renderer_pyrd import Renderer
from lib.common import constants
from lib.common.utils import strip_prefix_if_present, cam_crop2full, video_to_images
from lib.common.utils import estimate_focal_length
from lib.common.renderer_pyrd import Renderer
# from lib.yolov3_detector import HumanDetector
from lib.common.mocap_dataset import MocapDataset
# from lib.yolov3_dataset import DetectionDataset
from lib.common.imutils import process_image
from lib.common.utils import estimate_focal_length



from turtle import pos
import cv2


import numpy as np
import torch
from lib.config import get_config


# from pose_2D import detect_pose




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("--------------------------- 3D HPS estimation ---------------------------")


parser = argparse.ArgumentParser(description='PyTorch 3D Face Reconstruction')
parser.add_argument('--cfg', default='cfg/config.yaml', type=str, help='config file path')
opt = parser.parse_args()

cfg = get_config(opt.cfg)


kopt_cnn = KeypointRCNN()
model = CLIFF(cfg).to(device)
model = torch.compile(model)


checkpoint = torch.load(cfg.CKPT_DIR + f"/final_weights_{cfg.EXP_NAME}.pth", map_location="cuda:0")
model.load_state_dict(checkpoint['state_dict'])




model.eval()



def extract_bounding_box(points):
	x_coordinates, y_coordinates = zip(*points)

	return torch.tensor([min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)])


vid = cv2.VideoCapture(0)

# Setup the SMPL model
smpl_model = smplx.create(cfg.SMPL.SMPL_MODEL_DIR, "smpl").to(device)


focal_length = 800
img_w = 640
img_h = 480

renderer = Renderer(focal_length= 800, img_w=640, img_h=480, faces=smpl_model.faces,
                      
                            same_mesh_color=("video" == "video"))



# def detect_person(frame):
# 	results = yolov5(frame)
# 	final_results = results.pandas().xyxy[0].values.tolist()
# 	for result in final_results:
# 		x1, y1, x2, y2, conf, _,  cls = result
# 		if cls == "person":
# 			return torch.tensor([x1, y1, x2, y2])
			



c = 0
while True:
	start_time_full = time.time()
	ret, img_bgr = vid.read()
	# img_bgr = cv2.imread(path +  files[c])
	# c+=1
	#img_bgr = cv2.imread("/media/pranoy/Pranoy/mpi_inf_3dhp/S1/Seq1/imageFrames/all_images/frame_004921.jpg")
	# img_bgr = cv2.imread("images/frame_003809.jpg")
	draw_img = img_bgr.copy()
	# img_bgr = cv2.resize(img_bgr, (512, 512))
	


	# norm_img = (letterbox_image(img_bgr, (416, 416)))
	# norm_img = norm_img[:, :, ::-1].transpose((2, 0, 1)).copy()
	# norm_img = norm_img / 255.0

	# norm_img = torch.from_numpy(norm_img)
	# norm_img = norm_img.to(device).float()
	# norm_img = norm_img.unsqueeze(0)

	# dim = np.array([img_bgr.shape[1], img_bgr.shape[0]])
	# dim = torch.from_numpy(dim)
	# dim = dim.unsqueeze(0)
	# dim = dim.to(device)


	# detection_result = human_detector.detect_batch(norm_img, dim)

	# kpt_time_start = time.time()
	# kpt_results = detect_pose(img_bgr)# shape (18, 2)
	# scaled_keypoints = kpt_results["scaled_keypoints"]

	yolov5_time_start = time.time()
	img_rgb = img_bgr[:, :, ::-1]
	bbox = kopt_cnn.detect_pose(img_rgb.copy())["bbox"]
	bbox = bbox.to(device)

	print("YOLOV5 PIPE FPS: ", 1/(time.time() - yolov5_time_start))



	# extra_fps = time.time()
	# bbox = extract_bounding_box(scaled_keypoints).to(device)

	# print(bbox)
	# print("extra time: ", time.time() - extra_fps)



	cliff_time_Start = time.time()

	img_rgb = img_bgr[:, :, ::-1]
	img_h, img_w, _ = img_rgb.shape


	img_h, img_w, _ = img_bgr.shape
	img_h = torch.tensor([img_h]).float().to(device)
	img_w = torch.tensor([img_w]).float().to(device)

	focal_length = estimate_focal_length(img_h, img_w)
	# bbox = detection_result[0][1:5]



	norm_img, center, scale, crop_ul, crop_br, _ = process_image(cfg, img_rgb, bbox)


	
	center = center.unsqueeze(0).to(device)
	scale = scale.unsqueeze(0)
	focal_length = torch.tensor([focal_length]).to(device)







	pred_vert_arr = []
	cx, cy, b = center[:, 0], center[:, 1], scale * 200

	# print(cx, cy, b)

	
	bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
	# The constants below are used for normalization, and calculated from H36M data.
	# It should be fine if you use the plain Equation (5) in the paper.
	bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
	bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]


	norm_img = norm_img.unsqueeze(0)
	norm_img = norm_img.to(device)



	with torch.no_grad():
		
		pred_rotmat, pred_betas, pred_cam_crop = model(norm_img, bbox_info,  n_iter=3)
		

	

	# convert the camera parameters from the crop camera to the full camera

	# full_img_shape = torch.tensor([[img_h, img_w]]).float().to(device)
	full_img_shape = torch.stack((img_h, img_w), dim=-1)

	# full_img_shape = torch.stack((img_h, img_w), dim=-1)
	pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)



	pred_output = smpl_model(betas=pred_betas,
								body_pose=pred_rotmat[:, 1:],
								global_orient=pred_rotmat[:, [0]],
								pose2rot=False,
								transl=pred_cam_full)


	

	vertices = pred_output.vertices
	faces = smpl_model.faces
	joints = pred_output.joints
	print("CLIFF FPS: ", 1 / (time.time() - cliff_time_Start))




	start_time = time.time()
	front_view = renderer.render_front_view(vertices.cpu(), img_bgr)
	end_time = time.time()
	front_view = cv2.resize(front_view, (640, 480))

	print("RENDERING FPS: ", 1 / (end_time - start_time))

	
	end_time_full = time.time()
	print("FULL FPS: ", 1 / (end_time_full - start_time_full))


	cv2.imshow('image', front_view)

	cv2.waitKey(1)
	





 
	
	

