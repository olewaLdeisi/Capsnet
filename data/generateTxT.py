"""
-------------------------------------------------
   Author :        lin
   date：          2019/5/16 18:27
-------------------------------------------------
"""
__author__ = 'lin'

import os

print(len(os.listdir('./img')))
with open('./train.txt', 'r') as train_file:
    with open('./test.txt', 'r') as test_file:
        train_list = train_file.readlines()
        test_list = test_file.readlines()
        for i in train_list:
            if i in test_list:
                print(i)
import sys
sys.exit(0)

train_list = [i+'@0\n' for i in os.listdir('./train/narrow')] + [i+'@1\n' for i in os.listdir('./train/wide')]
test_list = [i+'@0\n' for i in os.listdir('./test/narrow')] + [i+'@1\n' for i in os.listdir('./test/wide')]

with open('./train.txt', 'w') as train_file:
    for i in train_list:
        train_file.write(i)

with open('./test.txt', 'w') as test_file:
    for i in test_list:
        test_file.write(i)

with open('./t.txt', 'w') as total_file:
    for i in train_list:
        total_file.write(i)
    for i in test_list:
        total_file.write(i)

print(len(train_list), len(test_list))