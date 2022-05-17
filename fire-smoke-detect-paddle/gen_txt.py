# coding:utf-8
import os

fire_folder = 'fire_smoke'
neg_folder = 'neg_pics'
def gen_txt(fire_folder, neg_folder):
    with open('/home/aistudio/dataset/test_list.txt', 'w') as f:
        for pic in os.listdir(fire_folder):
            line = os.path.join(fire_folder, pic)+ ' '+'1'+'\n'
            f.write(line)
        for pic in os.listdir(neg_folder):
            line = os.path.join(neg_folder, pic)+ ' '+'0'+'\n'
            f.write(line)

gen_txt(fire_folder, neg_folder)
