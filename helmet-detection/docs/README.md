```shell
nvidia-smi

cd helmet-detection
sudo docker run -ti --volume="$(pwd)":/app --rm yiluxiangbei/paddlehub:v1.0 bash

sudo docker build -t yiluxiangbei/all-paddle-hub:v1.0 .
sudo docker push yiluxiangbei/all-paddle-hub:v1.0

sudo docker run -it --rm yiluxiangbei/all-paddle-hub:v1.0 bash
hub serving start -c config/config.json --cors True

sudo docker build -t yiluxiangbei/all-paddle-hub:v1.1 .
sudo docker push yiluxiangbei/all-paddle-hub:v1.1

sudo docker run -it --name all-paddle-hub -p 8866:8866 --rm yiluxiangbei/all-paddle-hub:v1.1
sudo docker run -it --name all-paddle-hub -p 8866:8866 -d yiluxiangbei/all-paddle-hub:v1.1

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

http://49.232.6.131:8866/
http://49.232.6.131:8866/predict/fire-smoke-detect-paddle
http://49.232.6.131:8866/predict/helmet-detection-hub

docker rmi `docker images | grep none | awk '{print $3}'`
```

```
Downloading yolov3_darknet53_270e_coco.pdparams from https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams
https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams
https://aistudio.baidu.com/aistudio/datasetdetail/50329
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