```shell
cd oral-detection
sudo docker run -ti --volume="$(pwd)":/app --rm yiluxiangbei/paddlehub:v1.0 bash

# 1.0 no ENTRYPOINT
sudo docker build -t yiluxiangbei/all-paddle-hub:v1.0 .
sudo docker push yiluxiangbei/all-paddle-hub:v1.0

sudo docker run -it --rm yiluxiangbei/all-paddle-hub:v1.0 bash
hub serving start -c config/config.json --cors True

sudo docker build -t yiluxiangbei/all-paddle-hub:v1.1 .
sudo docker push yiluxiangbei/all-paddle-hub:v1.1

sudo docker run -it --name all-paddle-hub -p 8096:8866 --rm yiluxiangbei/all-paddle-hub:v1.1
sudo docker run -it --name all-paddle-hub -p 8096:8866 -d yiluxiangbei/all-paddle-hub:v1.1

sudo docker logs -f all-paddle-hub

sudo docker stop all-paddle-hub
sudo docker start all-paddle-hub
sudo docker rm all-paddle-hub

hub install chinese_ocr_db_crnn_server==1.1.2
hub run chinese_ocr_db_crnn_server --input_path "/PATH/TO/IMAGE"
hub serving start -m chinese_ocr_db_crnn_server
 
sudo docker run -ti --volume="$(pwd)":/app --rm yiluxiangbei/paddlehub:v1.0 bash
cd /app
python client.py
```

```shell
pip install paddlex
pip install paddlepaddle
https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/train#%E7%89%88%E6%9C%AC%E5%8D%87%E7%BA%A7
paddlex.det.transforms 2.0

!unzip data/data94616/口腔图像重要部位实例分割精度提升方案.zip
!mv 口腔图像重要部位实例分割精度提升方案 dataset
!unzip dataset/train.zip -d dataset
!unzip dataset/test.zip -d dataset

from PIL import Image

png_img = Image.open('dataset/train/images/00001.jpg')
png_img  # 展示真实图片

!pip install paddlex==1.3.10
!cp dataset/train/annotations/train.json dataset/train/annotations.json
!paddlex --split_dataset --format COCO --dataset_dir dataset/train --val_value 0.2

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.Normalize(),
    transforms.ResizeByShort(
        short_size=224, max_size=500), transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(
        short_size=224, max_size=500),
    transforms.Padding(coarsest_stride=32),
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-cocodetection
train_dataset = pdx.datasets.CocoDetection(
    data_dir='dataset/train/images',
    ann_file='dataset/train/train.json',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='dataset/train/images',
    ann_file='dataset/train/val.json',
    transforms=eval_transforms)
    
# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
# num_classes 需要设置为包含背景类的类别数，即: 目标类别数量 + 1
num_classes = len(train_dataset.labels) + 1

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/instance_segmentation.html#maskrcnn
model = pdx.det.MaskRCNN(num_classes=num_classes, backbone='ResNet50_vd', with_dcn=True)

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/instance_segmentation.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=120,
    train_dataset=train_dataset,
    train_batch_size=48,
    eval_dataset=eval_dataset,
    learning_rate=0.00125,
    warmup_steps=200,
    lr_decay_epochs=[20, 50, 80, 110],
    save_dir='output/mask_rcnn_hrnet_fpn',
    save_interval_epochs =3,
    pretrain_weights = "COCO",
    use_vdl=True)
    
!mkdir predict
import paddlex as pdx


model = pdx.load_model('output/mask_rcnn_hrnet_fpn/best_model')
result = model.predict('dataset/test/images/00000.jpg')

pdx.det.visualize('dataset/test/images/00000.jpg', result, save_dir='./predict')

from PIL import Image
import numpy as np

png_img = Image.open('dataset/test/images/00000.jpg')
png_img  # 展示真实图片

# 预测结果保存在predict/visualize_00000.jpg
png_img = Image.open('predict/visualize_00000.jpg')
png_img  # 展示真实图片


import os
import numpy as np
import pycocotools.mask as mask
import json


test_path = 'dataset/test/images/'
files = os.listdir(test_path)
files.sort()
# print(files[:5])

idx = 1
end_rs = []
for f in files:
    result = model.predict(os.path.join(test_path, f))

    # print(result)

    
    # print(result[0])
    for r in result:
        rs = {}
        rs["image_id"] = idx
        rs["bbox"] = r["bbox"]
        rs["score"] = r["score"]
        rs["category_id"] = r["category_id"]
        rs["segmentation"] = mask.encode(r['mask'])
        rs["segmentation"]["counts"] = str(rs["segmentation"]["counts"], encoding = "utf-8") 
        end_rs.append(rs)
    idx += 1
#         break
#     break

# print(end_rs)

with open("results.json",'w') as file_obj:
    json.dump(end_rs, file_obj)
```