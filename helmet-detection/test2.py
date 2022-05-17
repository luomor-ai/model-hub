import cv2
import paddlex as pdx
import numpy as np
import colorsys
import os

predictor = pdx.deploy.Predictor('/home/aistudio/output/inference_model/inference_model')
cap = cv2.VideoCapture('/home/aistudio/work/Industry.mp4')
save_dir = '/home/aistudio/frames'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
i = 1
det_rate = 20
save_dir = "/home/aistudio/frames/"
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if i % det_rate == 0:
            result = predictor.predict(frame)
            print(i)
            vis_img = pdx.det.visualize(frame, result, save_dir=save_dir)

            # 本地环境可以实时查看安全帽检测效果
            # cv2.imshow('hatdet', vis_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        i += 1
    else:
        break
cap.release()