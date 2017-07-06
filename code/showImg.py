import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from getPath import *
pardir = getparentdir()
train_data_path = pardir+'/data/train.csv'
imgsize = 28

def readData():
    data = pd.read_csv(train_data_path,encoding = 'utf-8')
    labels = data['label']
    columes = list(data.columns.values)
    pic_data = data[columes[1:]]
    length = len(pic_data)
    label_arr = []
    pic_arr = []
    for i in range(length):
        label = labels[i]
        pic = list(pic_data.iloc[i])
        label_arr.append(label)
        pic_arr.append(pic)
    pic_arr = np.array(pic_arr)    
    return label_arr,pic_arr

def showImg():
    label_arr,pic_arr = readData()
    for i in range(len(pic_arr)):
        img = np.reshape(pic_arr[i],[imgsize,imgsize])
        plt.title(label_arr[i])
        plt.imshow(img)
        plt.show()

if __name__=="__main__":
    showImg()
    
    