import cv2
import numpy as np
import sys
import json

import onnxruntime

from facechanger import utils, constants
from facechanger.transform import transform

# face alignment:
from alignment.retinaface import RetinaFace
from alignment.face_alignment import get_cropped_head_256
detector = RetinaFace("alignment/scrfd_2.5g_bnkps.onnx", provider=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"], session_options=None)

# kp detector:
from kp_detect.kps_student import KP
kp_detector = KP(model_path="kp_detect/kps_student.onnx", device="cuda")


def process_image(model, img, size, crop_scale):

    size = args.size

    bboxes, kpss = model.detect(img, (256, 256), det_thresh=0.6)
    if kpss is None or len(kpss) == 0:
        raise Exception("No face detected")
    
    assert len(kpss) != 0, "No face detected"
    aimg, mat = get_cropped_head_256(img, kpss[0], size=args.size, scale=crop_scale)
    
    return aimg, mat

def detect_features(img, kp_detector):
    """Detects facial features from a given image."""
    faces = [(0, 0, img.shape[1], img.shape[0])]  # Assuming the whole image as face region
    
    (x, y, w, h) = faces[0]

    # Crop and resize the face region for keypoint detection
    face_img = img[y:y+h, x:x+w]
    
    landmarks = kp_detector.get_kp(face_img)
    if args.size == 512:
        landmarks = landmarks * 2
            
    # Transform the landmarks to the coordinates in the original image
    landmark_list = [(int(point[0] * w / args.size + x), int(point[1] * h / args.size + y)) for point in landmarks]

    highest_y = np.array(landmark_list)[constants.LOWER_HEAD].min(axis=0)[1]

    for i in constants.LOWER_HEAD[1:-1]:
        #landmark_list.append((landmark_list[i][0], max(1,2*highest_y-landmark_list[i][1])))
        
        # alternative:
        landmark_list.append((landmark_list[i][0], int(max(1, highest_y - (highest_y - landmark_list[i][1]) *-0.6))))

    return np.array(landmark_list)

def draw_keypoints(image, keypoints):
    
    lower_head = keypoints[0:33]
    left_brow = keypoints[33:42]
    right_brow = keypoints[42:51]
    nose = keypoints[51:60]
    left_eye = keypoints[60:68]
    right_eye = keypoints[68:76]
    mouth = keypoints[76:96]
    
    # Draw face parts lines
    cv2.polylines(image, [lower_head], False, (255,255,255), 1)
    cv2.polylines(image, [left_brow], True, (0,255,0), 1)
    cv2.polylines(image, [right_brow], True, (0,255,0), 1)
    cv2.polylines(image, [nose], False, (255,0,255), 1)
    cv2.polylines(image, [left_eye], True, (0,0,255), 1)
    cv2.polylines(image, [right_eye], True, (0,0,255), 1)
    cv2.polylines(image, [mouth], True, (0,255,255), 1)    
    
    # Draw keypoints
    #for point in keypoints[:88]: # 98
    #    cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), args.size // 256)
    
    return image

def zoom(points, factor):
    """Zoom the points by the given factor, with the anchor at the center."""
    center = np.mean(points, axis=0)
    return (points - center) * factor + center

def get_new_features(features, filter):
    """Apply filter to features."""
    new_features = features.copy()

    for k, v in filter.items():
        indices = constants.INDICES[k]
        new_features[indices] = zoom(new_features[indices], v["zoom"]) + v["trans"]
    
    return np.array(new_features)

if __name__ == "__main__":

    args = utils.parse_args()
    crop_scale = args.crop
    
    # static mask for alignement
    r_mask = np.zeros((args.size, args.size), dtype=np.uint8)
    r_mask = cv2.rectangle(r_mask, (10, 10), (args.size - 10, args.size - 10), (255, 255, 255), -1)
    r_mask = cv2.cvtColor(r_mask, cv2.COLOR_GRAY2RGB)
    r_mask = cv2.GaussianBlur(r_mask, (11, 11), 0)
    r_mask = r_mask / 255
    
    # use existing filter
    ui_handler = utils.UserInputHandler(args.filter)

    # Set up OpenCV window
    windowname = "Press 'r' to reset, 'Enter' to continue..."
    cv2.namedWindow(windowname, cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(windowname, ui_handler.click)

    # Load the image
    img = cv2.imread(args.input)
    
    if img is None:
        raise Exception(f"Error loading image {args.input}")

    w, h = img.shape[1], img.shape[0]
    full_img = img.copy()
    
    cv2.resizeWindow(windowname, 512,512)


    img, matrix = process_image(detector, img, 256, crop_scale=crop_scale)
    inverse_matrix = cv2.invertAffineTransform(matrix)
    
    # Detect features
    features = detect_features(img, kp_detector)
    features_reset = np.copy(features)

    while True:
        if features is not None:
            # Draw keypoints on the image
            img_with_keypoints = draw_keypoints(img.copy(), features)
            
            # Get new features based on user input
            new_features = get_new_features(features, ui_handler.get_filter())
            ui_handler.features = new_features
            # Apply transformations to the image
            new_img = transform(img, features, new_features)
            
            # Draw new keypoints on the transformed image
            new_img_with_keypoints = draw_keypoints(new_img.copy(), new_features)
        else:
            img_with_keypoints = img
            new_img_with_keypoints = img

                
        cv2.imshow(windowname, new_img_with_keypoints)
        k = cv2.waitKey(1)

        if k == ord('r'):
            print("Resetting to default settings...")
            ui_handler.filter = constants.DEFAULT
            new_features = get_new_features(features, ui_handler.get_filter())
            new_img = transform(img, features, new_features)
            new_img_with_keypoints = draw_keypoints(new_img.copy(), new_features)
            ui_handler.features = new_features
                            
        if k == 13:  # 'Enter' key pressed to continue
            cv2.destroyAllWindows()
            new_img = transform(img, features, new_features)          
            new_img = cv2.warpAffine(new_img, inverse_matrix, (w, h))
            mask = cv2.warpAffine(r_mask, inverse_matrix, (w, h))            
            
            full_img = mask * new_img + (1 - mask) * (full_img)

            if args.output is not None:                
                cv2.imwrite(args.output,full_img)
            if args.save is not None:
                with open(args.save, "w+") as f:
                    json.dump(ui_handler.get_filter(), f, indent=1)
                            
            cv2.imshow("Result", full_img.astype(np.uint8))
            cv2.waitKey()

            break
    
    cv2.destroyAllWindows()
    sys.exit()
