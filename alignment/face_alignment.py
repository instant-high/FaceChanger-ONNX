import cv2
import numpy as np

    
def align_crop_256(img, landmark, size):
    template_ffhq = np.array(
	[
		[192.98138, 239.94708],
		[318.90277, 240.19366],
		[256.63416, 314.01935],
		[201.26117, 371.41043],
		[313.08905, 371.15118]
	])

    if size == 256:
        template_ffhq = template_ffhq /2
    if size == 512:
        template_ffhq *= (256 / size)
    
    matrix = cv2.estimateAffinePartial2D(landmark, template_ffhq, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    
    if size == 512:
        matrix = matrix * 2
    
    warped = cv2.warpAffine(img, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)
    return warped, matrix


def get_cropped_head_256(img, landmark, scale=1.4, size=512):
    center = np.mean(landmark, axis=0)
    landmark = center + (landmark - center) * scale
    return align_crop_256(img, landmark, size)
