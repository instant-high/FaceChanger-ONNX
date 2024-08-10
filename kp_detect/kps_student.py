import cv2
import onnxruntime
import onnx
import numpy as np

class KP:
    def __init__(self, model_path="kps_student.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        model = onnx.load(model_path)

        
    def get_kp(self, image):
    
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1)) / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
                
        kp = self.session.run(None,{'input':image})[0]
        
        kp = np.array(kp)[:98*2].reshape(-1,2)
        kp = kp * [256,256]

        return kp
