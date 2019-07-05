"""
-------------------------------------------------
   Author :        lin
   date：          2019/5/16 20:27
-------------------------------------------------
"""
__author__ = 'lin'
import random

with open('train.txt', 'r') as train_file:
    with open('train_new.txt', 'w') as new_train_file:
        true_img = []
        false_img = []
        data = train_file.readlines()
        for i in data:
            if i.strip().split('@')[-1] == '1':
                true_img.append(i)
            else:
                false_img.append(i)
        print('true_img', true_img)
        print('len:', len(true_img))
        print('false_img', false_img)
        print('len:', len(false_img))
        fill_num = len(false_img) - len(true_img)
        fill_list = random.choices(true_img, k=fill_num)

        true_img += fill_list
        for i in true_img:
            # print(i)
            new_train_file.write(i)
        for i in false_img:
            new_train_file.write(i)

