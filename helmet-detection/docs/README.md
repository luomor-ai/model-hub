```shell

```

```shell
!unzip  /home/aistudio/data/data50329/HelmetDetection.zip -d /home/aistudio/work/dataset
%cd /home/aistudio/work/dataset
%mv annotations Annotations
%mv images JPEGImages

!pip install paddlex
!paddlex --split_dataset --format voc --dataset_dir /home/aistudio/work/dataset --val_value 0.15

# 单卡训练
%cd /home/aistudio/
!python /home/aistudio/code/train.py

# 四卡训练
%cd /home/aistudio/
!export CUDA_VISIBLE_DEVICES=0,1,2,3
!python -m paddle.distributed.launch --gpus 0,1,2,3 /home/aistudio/code/train.py

!python code/infer.py

!paddlex --export_inference --model_dir=/home/aistudio/models/yolov3_darknet53 \
         --save_dir=output/inference_model --fixed_input_shape=[480,480]
         
pip install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple
!hub convert --model_dir /home/aistudio/output/inference_model/inference_model --module_name helmet_hub
!hub install helmet_hub_1652166294.6532774/helmet_hub.tar.gz
!hub serving start -m helmet_hub
```