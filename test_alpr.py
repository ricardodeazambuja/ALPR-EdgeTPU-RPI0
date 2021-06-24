from io import BytesIO
from time import monotonic
import numpy as np
from PIL import Image
from picamera import PiCamera

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
import tflite_runtime.interpreter as tflite

SCORE_MIN = 0.5 # used by the object (licence plate) detector

# This LPRNet was trained on Chinese license plates:
char2value = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '<Anhui>': 10, '<Beijing>': 11, '<Chongqing>': 12, '<Fujian>': 13, '<Gansu>': 14, '<Guangdong>': 15, '<Guangxi>': 16, '<Guizhou>': 17, '<Hainan>': 18, '<Hebei>': 19, '<Heilongjiang>': 20, '<Henan>': 21, '<HongKong>': 22, '<Hubei>': 23, '<Hunan>': 24, '<InnerMongolia>': 25, '<Jiangsu>': 26, '<Jiangxi>': 27, '<Jilin>': 28, '<Liaoning>': 29, '<Macau>': 30, '<Ningxia>': 31, '<Qinghai>': 32, '<Shaanxi>': 33, '<Shandong>': 34, '<Shanghai>': 35, '<Shanxi>': 36, '<Sichuan>': 37, '<Tianjin>': 38, '<Tibet>': 39, '<Xinjiang>': 40, '<Yunnan>': 41, '<Zhejiang>': 42, '<police>': 43, 'A': 44, 'B': 45, 'C': 46, 'D': 47, 'E': 48, 'F': 49, 'G': 50, 'H': 51, 'I': 52, 'J': 53, 'K': 54, 'L': 55, 'M': 56, 'N': 57, 'O': 58, 'P': 59, 'Q': 60, 'R': 61, 'S': 62, 'T': 63, 'U': 64, 'V': 65, 'W': 66, 'X': 67, 'Y': 68, 'Z': 69, '_': 70}

# Generates a dictionary to revert:
value2char = {v:k for k,v in char2value.items()}

#
# This system is divided into two models that were
# compiled to share the cache memory (edgetpu_compiler -s model_dense2conv_mod.tflite ssdlite_ocr.tflite)
# Therefore, two tflite interpreter objects will be created.
# Original models from: https://github.com/GreenWaves-Technologies/licence_plate_recognition

# Interpreter for the object detector
model_file = "ssdlite_ocr_edgetpu.tflite"
device = [] # I have only one USB accelerator...
delegate = tflite.load_delegate(EDGETPU_SHARED_LIB,{'device': device[0]} if device else {})
tflite_interpreter_obj_detector = tflite.Interpreter(model_path=model_file, 
                                 experimental_delegates=[delegate])
tflite_interpreter_obj_detector.allocate_tensors()
tflite_interpreter_obj_detector.invoke()


# Interpreter for the licence plate OCR
model_file = "lprnet_mod_edgetpu.tflite"
device = [] # I have only one USB accelerator...
delegate = tflite.load_delegate(EDGETPU_SHARED_LIB,{'device': device[0]} if device else {})
tflite_interpreter_plate_ocr = tflite.Interpreter(model_path=model_file, 
                                 experimental_delegates=[delegate])
tflite_interpreter_plate_ocr.allocate_tensors()
tflite_interpreter_plate_ocr.invoke()


image_width, image_height = (320, 240)
with PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 30
    
    stream = BytesIO()
    for foo in camera.capture_continuous(stream,
                                         format='rgb',
                                         use_video_port=True,
                                         resize=(320, 240)):
        init_time = monotonic()
        stream.truncate()
        stream.seek(0)
        image_raw = np.frombuffer(stream.getvalue(), dtype=np.uint8) # Images are never saved to disk
        image_np = np.reshape(image_raw, [  1, 240, 320,   3])
        tflite_interpreter_obj_detector.set_tensor(7, image_np) # The number 7 is the index of the input tensor,
                                                                # Check tflite_interpreter_obj_detector.get_input_details()
        tflite_interpreter_obj_detector.invoke()
        
        scores = tflite_interpreter_obj_detector.get_tensor(3)[0] # The number 3 is the index of the tensor with scores
                                                                  # Check tflite_interpreter_obj_detector.get_output_details()
        score_max_idx = scores.argmax()
        score_max = scores[score_max_idx]
        if score_max >= SCORE_MIN:
            box = tflite_interpreter_obj_detector.get_tensor(1)[0][score_max_idx] # The number 1 is the index of the tensor with bounding boxes
                                                                                  # Check tflite_interpreter_obj_detector.get_output_details()
            ymin = int(box[0] * image_height)
            xmin = int(box[1] * image_width)
            ymax = int(box[2] * image_height)
            xmax = int(box[3] * image_width)

            plate_np = np.asarray(Image.fromarray(image_np.reshape((240,320,3))).crop((xmin,ymin,xmax,ymax)).resize((94,24))).reshape((1,24,94,3))
            tflite_interpreter_plate_ocr.set_tensor(1, plate_np) # The number 1 is the index of the input tensor
                                                                 # Check tflite_interpreter_plate_ocr.get_input_details()
            tflite_interpreter_plate_ocr.invoke()
            mod1_output = tflite_interpreter_plate_ocr.get_tensor(0)[0]

            plate_characters = " "
            output_characters = ["_"]
            for c in mod1_output.argmax(axis=1)[7:]:
              output_characters.append(str(value2char[c]))
              # Ignores if repeated or "_"
              if (output_characters[-1] == output_characters[-2]) or (output_characters[-1] == "_"):
                continue
              plate_characters += str(output_characters[-1])

            print(f"[{1/(monotonic()-init_time):.2f}Hz] Plate found (score:{score_max:.2f}): {plate_characters}")
        else:
            print(f"[{1/(monotonic()-init_time):.2f}Hz] No plates with score >= {SCORE_MIN:.2f} - Max score: {score_max:.2f}")
