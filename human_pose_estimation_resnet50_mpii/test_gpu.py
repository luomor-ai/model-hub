import cv2
import paddlehub as hub
import math
from matplotlib import pyplot as plt
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
% matplotlib
inline


def countYwqz():
    pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")

    flag = False
    count = 0
    num = 0
    all_num = []
    flip_list = []
    fps = 60
    # 可选择web视频流或者文件
    file_name = 'work/ywqz.mp4'
    cap = cv2.VideoCapture(file_name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out后期可以合成视频返回
    out = cv2.VideoWriter(
        'output.mp4',
        fourcc,
        fps,
        (width, height))

    while cap.isOpened():
        success, image = cap.read()
        # print(image)
        if not success:
            break
        image_height, image_width, _ = image.shape
        # print(image_height, image_width)

        image.flags.writeable = False
        results = pose_estimation.keypoint_detection(images=[image], visualization=True, use_gpu=True)

        flip = results[0]['data']['head_top'][1]  # 获取头部的y轴坐标值
        flip_list.append(flip)
        all_num.append(num)
        num += 1

    # 写入视频
    img_root = "output_pose/"
    # 排序，不然是乱序的合成出来
    im_names = os.listdir(img_root)
    im_names.sort(key=lambda x: int(x.replace("ndarray_time=", "").split('.')[0]))
    for im_name in range(len(im_names)):
        img = img_root + str(im_names[im_name])
        print(img)
        frame = cv2.imread(img)
        out.write(frame)
    out.release()

    return all_num, flip_list


def get_count(x, y):
    count = 0
    flag = False
    count_list = [0]  # 记录极值的y值
    for i in range(len(y) - 1):
        if y[i] <= y[i + 1] and flag == False:
            continue
        elif y[i] >= y[i + 1] and flag == True:
            continue
        else:
            # 防止附近的轻微抖动也被计入数据
            if abs(count_list[-1] - y[i]) > 100 or abs(count_list[-1] - y[i - 1]) > 100 or abs(
                    count_list[-1] - y[i - 2]) > 100 or abs(count_list[-1] - y[i - 3]) > 100 or abs(
                    count_list[-1] - y[i + 1]) > 100 or abs(count_list[-1] - y[i + 2]) > 100 or abs(
                    count_list[-1] - y[i + 3]) > 100:
                count = count + 1
                count_list.append(y[i])
                print(x[i])
                flag = not flag
    return math.floor(count / 2)


if __name__ == "__main__":
    x, y = countYwqz()

    plt.figure(figsize=(8, 8))
    count = get_count(x, y)
    plt.title(f"point numbers: {count}")
    plt.plot(x, y)
    plt.show()