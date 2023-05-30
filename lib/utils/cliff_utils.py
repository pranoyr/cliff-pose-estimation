import torch
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from lib.augmentation.geometry import batch_rodrigues


def convert_to_angle_axis(pred_pose, i):
	pred_rotmat_hom = torch.cat([pred_pose.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
				device="cpu").view(1, 3, 1).expand(1 * i, -1, -1)], dim=-1)
	pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(1, -1)[0]
	# pred_pose = pred_pose.view(-1, 3)
	return pred_pose

def convert_to_rotmat(pred_pose, i):
	pred_pose = batch_rodrigues(pred_pose.view(-1,3)).view(-1, i, 3, 3)
	return pred_pose[0]



def perspective_projection(points, rotation, translation,
						   focal_length, camera_center):
	"""
	This function computes the perspective projection of a set of points.
	Input:
		points (bs, N, 3): 3D points
		rotation (bs, 3, 3): Camera rotation
		translation (bs, 3): Camera translation
		focal_length (bs,) or scalar: Focal length
		camera_center (bs, 2): Camera center
	"""
	batch_size = points.shape[0]
	K = torch.zeros([batch_size, 3, 3], device=points.device)
	K[:,0,0] = focal_length
	K[:,1,1] = focal_length
	K[:,2,2] = 1.
	K[:,:-1, -1] = camera_center



	# Transform points
	points = torch.einsum('bij,bkj->bki', rotation, points)
	#points = points + translation.unsqueeze(1)

	# Apply perspective distortion
	projected_points = points / points[:,:,-1].unsqueeze(-1)

	# Apply camera intrinsics
	projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

	return projected_points[:, :, :-1]


def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)


    # full_cam = torch.stack([crop_cam[:,1],
    #                               crop_cam[:,2],
    #                               2*focal_length/(512 * crop_cam[:,0] +1e-9)],dim=-1)
    return full_cam



def get_landmarks(face_proj):
		"""
		Return:
			face_lms         -- torch.tensor, size (B, 68, 2)

		Parameters:
			face_proj       -- torch.tensor, size (B, N, 2)
			
		"""  

		body_idx = np.squeeze(np.arange(24)) 
		body_idx = torch.from_numpy(body_idx).cuda().long()
		return face_proj[:, body_idx]
		# return face_proj