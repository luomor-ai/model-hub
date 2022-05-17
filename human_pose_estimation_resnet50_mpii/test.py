import cv2
import paddlehub as hub

pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")

image1=cv2.imread('work/1.png') # 坐直
image2=cv2.imread('work/2.png') # 全躺
image3=cv2.imread('work/3.png') # 中间状态
results = pose_estimation.keypoint_detection(images=[image1,image2,image3], visualization=True)

# 打印关键点
print(results[0]['data'])
