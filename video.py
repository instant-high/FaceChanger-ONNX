import cv2
import numpy as np
import sys
import json
import subprocess, platform, shutil, os

import onnxruntime

from facechanger import utils, constants
from facechanger.transform import transform

from tqdm import tqdm

# face alignment:
from alignment.retinaface import RetinaFace
from alignment.face_alignment import get_cropped_head_256
face_detector = RetinaFace("alignment/scrfd_2.5g_bnkps.onnx", provider=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"], session_options=None)

device = 'cuda'

# kp detector:
from kp_detect.kps_student import KP
kp_detector = KP(model_path="kp_detect/kps_student.onnx", device=device) #kps_student 98kp


video_landmark_data = []
           
def process_video(model, img, size, crop_scale):

    size = args.size
    
    # detection model input 256
    bboxes, kpss = model.detect(img, (256, 256), det_thresh=0.6)
    aimg, mat = get_cropped_head_256(img, kpss[0], size=args.size, scale=crop_scale)
    
    return aimg, mat
    
def detect_features(img, kp_detector, smooth_sigma=1):
    faces = [(0, 0, img.shape[1], img.shape[0])]
    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]
    
    # kp_detection model input 256
    face_img = cv2.resize(face_img,(256, 256))
    landmarks = kp_detector.get_kp(face_img)
    # for 512 processing
    if args.size == 512:
        landmarks = landmarks * 2

    landmark_list = [(int(point[0] * w / args.size + x), int(point[1] * h / args.size + y)) for point in landmarks]
    highest_y = np.array(landmark_list)[constants.LOWER_HEAD].min(axis=0)[1]
    
    for i in constants.LOWER_HEAD[1:-1]:
        #landmark_list.append((landmark_list[i][0], max(1, 2 * highest_y - landmark_list[i][1])))
        
        #alternative:
        landmark_list.append((landmark_list[i][0], int(max(1, highest_y - (highest_y - landmark_list[i][1]) * -0.6))))
       
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
    # 96,97 # 88 - 95   
    #for point in keypoints[:88]: # 98
    #    cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), args.size // 256)
    
    return image
    
def zoom(points, factor):
    center = np.mean(points, axis=0)
    return (points - center) * factor + center

def get_new_features(features, filter):
    new_features = features.copy()
    for k, v in filter.items():
        indices = constants.INDICES[k]
        new_features[indices] = zoom(new_features[indices], v["zoom"]) + v["trans"]

    return np.array(new_features)

if __name__ == "__main__":

    args = utils.parse_args()
    crop_scale = args.crop
    
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.endpos == 0:
        args.endpos = total_frames
            
    cutout = args.endpos - args.startpos # frames to process from cut in -> cut out
    audio_pos = (args.startpos/fps) # for merging original video audiotrack from startpos
        
    cap.set(1,args.startpos)
        
    # rescale input video:        
    scale  = int(args.scale)/10
    w = int(w * scale)
    h = int(h * scale)
    if w %2 !=0 : w = w - 1
    if h %2 !=0 : h = h - 1

    if args.output is not None:
        if args.audio:
            out = cv2.VideoWriter('temp.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))
        else:
            out = cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))
   

    # static mask for alignment    
    r_mask = np.zeros((args.size, args.size), dtype=np.uint8)
    r_mask = cv2.rectangle(r_mask, (10, 10), (args.size - 10, args.size - 10), (255, 255, 255), -1)
    r_mask = cv2.cvtColor(r_mask, cv2.COLOR_GRAY2RGB)
    r_mask = cv2.GaussianBlur(r_mask, (11, 11), 0)
    r_mask = r_mask / 255

    
    success, full_img = cap.read()
    full_img = cv2.resize(full_img,(w, h))
    
    if not success:
        raise Exception("Error while reading the first frame")
    
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    
    first_frame_processed = False
    features_copied = False
    
    # crop specific face region:
    print ("")
    print ("Select region of face to be changed")
    print ("")

    showCrosshair = False
    show_cropped = full_img
         
    roi = cv2.selectROI("Select region of face to be changed", show_cropped,showCrosshair)

    if roi == (0, 0, 0, 0):
        roi = (0, 0, w, h)
        face_region = full_img
    else:
        roiw=roi[2]
        roih=roi[3]
        if roi[2] %2 !=0 : roiw=(roi[2])-1
        if roi[3] %2 !=0 : roih=(roi[3])-1
        roi = (roi[0],roi[1],roiw,roih)			    
        face_region = full_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
      
    cv2.destroyAllWindows()
 
    

    # face changer window:
    ui_handler = utils.UserInputHandler(args.filter)       
    windowname = "Press 'r' to reset, 'Enter' to continue..."
    cv2.namedWindow(windowname, cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(windowname, ui_handler.click)
    cv2.resizeWindow(windowname, 512, 512)
    
    for index in tqdm(range(cutout)):
        success, full_img = cap.read()
  
        if success:
            full_img = cv2.resize(full_img,(w, h))
            ori_frame = full_img.copy()
            (full_h, full_w) = full_img.shape[:2]
                        
            # crop specific face region:
            face_region = full_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            (region_h,region_w) = face_region.shape[:2]
            
            try:
                img, matrix = process_video(face_detector, face_region, 256, crop_scale=crop_scale)
                #cv2.imshow("img",img)
            except:
                matrix = None
                pass

            features = detect_features(img, kp_detector) if img is not None else None

            if not features_copied:
                features_reset = np.copy(features)
                features_copied = True

            while not first_frame_processed:
                if features is not None:
                    img_with_keypoints = draw_keypoints(img.copy(), features)
                    new_features = get_new_features(features, ui_handler.get_filter())
                    ui_handler.features = new_features
                    new_img = transform(img, features, new_features)
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
                    
                if k == 13:
                    first_frame_processed = True
                    cv2.destroyAllWindows()
                    cap.set(1,args.startpos)
                    if args.save is not None:
                        with open(args.save, "w+") as f:
                            json.dump(ui_handler.get_filter(), f, indent=1)
                        
            if features is not None:
                new_features = get_new_features(features, ui_handler.get_filter())
                ui_handler.features = new_features
                new_img = transform(img, features, new_features)
                #new_img = draw_keypoints(new_img.copy(), new_features)
                cv2.imshow("new",new_img)
                
            else:
                new_img = img

            if matrix is not None:

                # opt face enhancer here...
                            
                inverse_matrix = cv2.invertAffineTransform(matrix)
                
                new_img = cv2.warpAffine(new_img, inverse_matrix, (region_w, region_h))
                mask = cv2.warpAffine(r_mask, inverse_matrix, (region_w, region_h))
                img = mask * new_img + (1 - mask) * (face_region)
                
                # insert specific face region back to full frame:
                ori_frame[int(roi[1]):int(roi[1])+ region_h, int(roi[0]):int(roi[0])+ region_w] = img
            else:
                ori_frame = img
            
            ori_frame = cv2.resize(ori_frame, (w, h))
        
            cv2.imshow("Result. Press 'Esc' to stop", ori_frame.astype(np.uint8))
            
            if args.output is not None:  
                out.write(ori_frame.astype(np.uint8))

            k = cv2.waitKey(1)
            if k == 27:
                break
                           
        else:
            break

    cap.release()
    if args.output is not None:
        out.release()
        if args.audio:
            result = 0
            
            command = 'ffmpeg.exe -y -vn -ss ' + str(audio_pos) + ' -i ' + '"' + args.input + '"' + ' -an -i ' + 'temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + args.output + '"'
            result = subprocess.call(command, shell=platform.system() != 'Windows')
            
            if result == 0:
                os.remove('temp.mp4')
            else:
                print("Input file has no audio stream")
                shutil.copyfile('temp.mp4', args.output)
                os.remove('temp.mp4')
               
    cv2.destroyAllWindows()
    sys.exit()
