import numpy as np
import paddlex as pdx
from paddlex import transforms as T

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/paddlex/cv/transforms/operators.py
train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=-1), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[384, 416, 448, 480, 512, 544, 576, 608, 640, 672],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=608, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/paddlex/cv/datasets/voc.py
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/work/dataset',
    file_list='/home/aistudio/work/dataset/train_list.txt',
    label_list='/home/aistudio/work/dataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/work/dataset',
    file_list='/home/aistudio/work/dataset/val_list.txt',
    label_list='/home/aistudio/work/dataset/labels.txt',
    transforms=eval_transforms,
    shuffle=False)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/tree/release/2.0.0/tutorials/train#visualdl可视化训练指标
num_classes = len(train_dataset.labels)
model = pdx.det.PPYOLOv2(
    num_classes=num_classes,
    backbone='ResNet50_vd_dcn',
    label_smooth=True,
    use_iou_aware=True,
    ignore_threshold=0.6)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/paddlex/cv/models/detector.py
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=200,  # 训练轮次
    train_dataset=train_dataset,  # 训练数据
    eval_dataset=eval_dataset,  # 验证数据
    train_batch_size=64,  # 批大小
    pretrain_weights='COCO',  # 预训练权重，刚开始训练的时候取消该注释，注释resume_checkpoint
    learning_rate=0.005 / 12,  # 学习率
    warmup_steps=500,  # 预热步数
    warmup_start_lr=0.0,  # 预热起始学习率
    save_interval_epochs=5,  # 每5个轮次保存一次，有验证数据时，自动评估
    lr_decay_epochs=[85, 135],  # step学习率衰减
    save_dir='output/PPYOLOv2_ResNet50_vd_dcn',  # 保存路径
    use_vdl=True)  # 其用visuadl进行可视化训练记录
