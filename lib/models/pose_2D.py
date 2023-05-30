import cv2
import numpy as np
import torch
import torchvision
import numpy as np

#-- person keypoints detection
# tested, not working well
class KeypointRCNN(object):
    ''' Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.
    Ref: https://pytorch.org/docs/stable/torchvision/models.html#keypoint-r-cnn
        'nose', - 0
        'left_eye', - 1
        'right_eye', - 2
        'left_ear', - 3
        'right_ear', 	 - 4
        'left_shoulder',	 - 5
        'right_shoulder', - 6
        'left_elbow', - 7
        'right_elbow', - 8
        'left_wrist', - 9
        'right_wrist', - 10
        'left_hip', - 11
        'right_hip', - 12
        'left_knee', - 13
        'right_knee', - 14
        'left_ankle', - 15
        'right_ankle' - 16
    '''
    def __init__(self, device='cuda:0'):  
    
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def run(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H
            labels (Int64Tensor[N]): the class label for each ground-truth box
            keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the format [x, y, visibility], where visibility=0 means that the keypoint is not visible.
        ''' 
     
        prediction = self.model(input.to(self.device))[0]
        # 
        kpt = prediction['keypoints'][0].cpu().numpy()
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
        bbox = torch.tensor([left, top, right, bottom])
        return bbox, torch.from_numpy(kpt)




    def detect_pose(self, img_rgb):
        results = {}
        #convert to tensor
        img = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
        img = img.unsqueeze(0)
        bbox, kpt = self.run(img)

        scaled_keypoints = kpt[:,:-1]
        normalised_keypoints = scaled_keypoints.clone()


        img_w = img_rgb.shape[1]
        img_h = img_rgb.shape[0]
        # normalize kpt with respect to image size
        normalised_keypoints[:,0] = scaled_keypoints[:,0] / img_w
        normalised_keypoints[:,1] = scaled_keypoints[:,1] / img_h


        results["normalised_keypoints"] = normalised_keypoints
        results["scaled_keypoints"] = scaled_keypoints
        results["bbox"] = bbox

        return results


