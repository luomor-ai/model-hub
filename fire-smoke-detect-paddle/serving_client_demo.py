# coding: utf8
import requests
import json
import cv2
import base64


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


if __name__ == '__main__':
    # 获取图片的base64编码格式
    img1 = cv2_to_base64(cv2.imread("fire_smoke/images/fire_000001.jpg"))
    img2 = cv2_to_base64(cv2.imread("fire_smoke/images/smoke_000001.jpg"))
    data = {'images': [img1, img2]}
    # 指定content-type
    headers = {"Content-type": "application/json"}
    # 发送HTTP请求
    url = "http://49.232.6.131:8866/predict/fire-smoke-detect-paddle"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])

    img = cv2.imread("fire_smoke/images/fire_000001.jpg")
    for j in range(len(r.json()["results"][0])):
        if(r.json()["results"][0][j]["score"] < 0.09):
            continue
        x, y, w, h = r.json()["results"][0][j]["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        x2, y2 = x + w, y + h
        # object_name = annos[j][""]
        img = cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), thickness=2)
        img = cv2.putText(img, r.json()["results"][0][j]["category"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        img_name = "fire_smoke/result.jpg"
        cv2.imwrite(img_name, img)