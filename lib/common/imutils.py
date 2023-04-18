# Copyright (c) 2019, University of Pennsylvania, Max Planck Institute for Intelligent Systems
# This script is borrowed and extended from SPIN

import cv2
import torch
import numpy as np
from torch.nn import functional as F
from scipy.ndimage.interpolation import rotate
import scipy
import numpy as np



def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def crop(img, center, scale, res, rot=0):
    """
    Crop image according to the supplied bounding box.
    res: [rows, cols]
    """
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)


    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    # new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

    if not rot == 0:
        # Remove padding
        new_img = rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)    

    return new_img, ul, br


def bbox_from_detector(cfg, bbox, rescale=1.1):
    """
    Get center and scale of bounding box from bounding box.
    The expected format is [min_x, min_y, max_x, max_y].
    """
    # center
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = torch.tensor([center_x, center_y])

    # scale
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    aspect_ratio = cfg.DATA.CROP_IMG_HEIGHT / cfg.DATA.CROP_IMG_WIDTH
    bbox_size = max(bbox_w * aspect_ratio, bbox_h)
    scale = bbox_size / 200.0
    # adjust bounding box tightness
    scale *= rescale
    return center, scale


def process_image(cfg, orig_img_rgb, bbox, rot=0, flip=0, train=False, pn=[]):
    """
    Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    crop_height = cfg.DATA.CROP_IMG_HEIGHT
    crop_width = cfg.DATA.CROP_IMG_WIDTH
    try:
        center, scale = bbox_from_detector(cfg, bbox)
    except Exception as e:
        print("Error occurs in person detection", e)
        # Assume that the person is centered in the image
        height = orig_img_rgb.shape[0]
        width = orig_img_rgb.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width * crop_height / float(crop_width)) / 200.

    img, ul, br = crop(orig_img_rgb, center, scale, (crop_height, crop_width), rot=rot)

    if flip:
        img = np.fliplr(img)

    crop_img = img.copy()

    if train:
        img[:,:,0] = np.minimum(255.0, np.maximum(0.0, img[:,:,0]*pn[0]))
        img[:,:,1] = np.minimum(255.0, np.maximum(0.0, img[:,:,1]*pn[1]))
        img[:,:,2] = np.minimum(255.0, np.maximum(0.0, img[:,:,2]*pn[2]))

    img = img / 255.
    mean = np.array(cfg.DATA.IMG_NORM_MEAN, dtype=np.float32)
    std = np.array(cfg.DATA.IMG_NORM_STD, dtype=np.float32)
    norm_img = (img - mean) / std
    norm_img = np.transpose(norm_img, (2, 0, 1))

    
    # convert to torch tensor
    norm_img = torch.from_numpy(norm_img).float()
    return norm_img, center, scale, ul, br, crop_img


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)
