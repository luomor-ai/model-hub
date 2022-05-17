# %matplotlib inline
import matplotlib.pyplot as plt # plt 用于显示图片
import numpy as np
import cv2
import os

save_dir='/home/aistudio/'
pdx.det.visualize(img_path, result, save_dir=save_dir)

path,filename = os.path.split(img_path)
output_path = os.path.join(save_dir,"visualize_"+filename)

pic = cv2.imread(output_path)
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
plt.imshow(pic)
plt.axis('off') # 不显示坐标轴
plt.show()