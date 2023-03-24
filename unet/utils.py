import cv2
import numpy as np



def correct_skew(img, src_pts = [], dst_pts =[]):
	# source_pts: from board.png
	

	src_pts =   [[89,68],
				[552,71],
				[98,364],
				[556,335]]

	# src_pts =   [[141,90],
	# 			[494,100],
	# 			[137,312],
	# 			[496,296]]

	# dest_pnts: from back_cam.png
	dst_pts =[[0,0] ,
				[640,0],
				[0,380],
				[640,380]]

	src_pts = np.float32(src_pts)
	dst_pts = np.float32(dst_pts)
	# dst_pts = np.float32(dst_pts)*5 +np.array((300,40))
	M, status = cv2.findHomography(src_pts, dst_pts)
	dst = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]))
	return dst




#
