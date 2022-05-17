import base64
import numpy as np
import cv2


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

def base64_to_np(b64tuple):
    data, shape = b64tuple
    data = base64.b64decode(data.encode('utf8'))
    data = np.frombuffer(data, np.float32).reshape(shape)
    return data

decoded_label_map = base64_to_cv2(label_map)
decoded_score_map = base64_to_np(score_map)

decoded_mask = base64_to_cv2(mask)