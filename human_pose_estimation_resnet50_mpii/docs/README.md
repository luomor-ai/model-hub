```shell
!pip install -U pip --user > log.log
!pip install -U paddlehub > log.log

!pip list |grep paddle

!hub install human_pose_estimation_resnet50_mpii > log.log
!hub list|grep human
```