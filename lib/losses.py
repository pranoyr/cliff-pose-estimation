
from torch.nn.functional import mse_loss



def compute_keypoint_loss(projected_keypoints_2d, target_landmarks):
    loss = mse_loss(projected_keypoints_2d, target_landmarks)
    return loss

def compute_beta_loss(beta_params, target_beta_params, is_params):
    beta_params = beta_params[is_params==1]
    target_beta_params = target_beta_params[is_params==1]
    loss = mse_loss(beta_params, target_beta_params)
    return loss

def compute_pose_loss(pose_params, target_pose_params, is_params):
    pose_params = pose_params[:, 1:][is_params==1]
    target_pose_params = target_pose_params[is_params==1]
    loss = mse_loss(pose_params, target_pose_params)
    return loss