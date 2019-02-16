import random

with open("./data/train.txt") as f:
    imglist = f.readlines()
    imgTrue = [i  for i in imglist if '1' in i.split('@')[-1]]
    imgTrueNew = imgTrue + imgTrue
    imgFalse = [i for i in imglist if '0' in i.split('@')[-1]]
    addNum = len(imgFalse) - len(imgTrueNew)
    addList = random.sample(imgTrue,addNum)
    imgTrue = imgTrueNew + addList
    with open('./data/train_new.txt','w') as f2:
        writeList = imgFalse+imgTrue
        f2.writelines(writeList)
    print(1)