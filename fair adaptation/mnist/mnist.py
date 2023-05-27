

import torch
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random

import numpy as np
import os
train_data = datasets.MNIST(
            root="./mnist/",
            train=True,
            transform=transforms.ToTensor(),
            download=True)

test_data = datasets.MNIST(
            root="./mnist/",
            train=False,
            transform=transforms.ToTensor(),
            download=True)
#print(train_data[0])


std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
batch_size=100
#print(img.shape)
'''
X_train = np.expand_dims(X_train, -1)
#X_test = np.expand_dims(X_test, -1)
X_train = np.concatenate([X_train, X_train, X_train], axis=3)
#X_test = np.concatenate([X_test, X_test, X_test], axis=3)
Y_train = np.array(Y_train).astype(np.int32)
#y_test = np.array(y_test).astype(np.int32)'''

#print(train_Y[0])
#plt.imshow(img)
#plt.show()
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
'''X_train = []
Y_train = []
for x, y in train_data:
    X_train.append(x.numpy())
    Y_train.append(y)
Y_train=np.array(Y_train)
print(Y_train.shape)
Y_train=torch.tensor(Y_train)
print(Y_train.shape)'''
n=100
k=10
pa = np.random.dirichlet(np.ones(10), size=100)
px= np.random.dirichlet(np.ones(10), size=100)
print(pa[0])
print(px[0])
def get_colorindex(img):
    color=[[0, 0, 128],[0, 128,0],[128, 0, 0],[0, 128, 128],[128, 0, 128],
            [128, 128, 0],[128, 128, 128],[255, 0, 0],[0, 255, 0],[0, 255, 255]]

def choose_color(index):
    color=[[0, 0, 128],[0, 128,0],[128, 0, 0],[0, 128, 128],[128, 0, 128],
            [128, 128, 0],[128, 128, 128],[255, 0, 0],[0, 255, 0],[0, 255, 255]]
    #print(pa[index])
    color_index=np.random.choice(10,1,p=pa[index])
    #print(color_index)
    return color[color_index[0]],color_index

#对一组图染色
def color_mnist(batch_img,index):
    color=[[0, 0, 128],[0, 128,0],[128, 0, 0],[0, 128, 128],[128, 0, 128],
            [128, 128, 0],[128, 128, 128],[255, 0, 0],[0, 255, 0],[0, 255, 255]]
    color_pro=px[index]
    color_num=[]
    for i in color_pro:
        color_num.append(round(batch_size*i))
    #print(sum(color_num))
    #print(color_num)
    color_num=color_num[:-1]
    #print(color_num)
    #print(sum(color_num))
    color_num_last=batch_size-sum(color_num)
    #print(color_num_last)
    color_num.append(color_num_last)
    #print(color_num)
    color_num=np.array(color_num)
    ini_color_index=[]
    t=0
    for num,a_color in zip(color_num,color):
        
        for i in range (num):
            ini_color_index.append(t)
            for j in range(0, 28):
                for k in range(0, 28):
                    if batch_img[i][j][k].tolist() != [.5, .5, .5]:
                        batch_img[i][j][k] = torch.tensor(np.array(a_color)).double()
        t=t+1
    return color_pro,ini_color_index


def random_unit(p: float):
    if p == 0:
        return False
    if p == 1:
        return True

    R = random.random()
    if R < p:
        return True
    else:
        return False

class MyDataset(Dataset):
    def __init__(self, train_data):
        X_train = []
        Y_train = []
        for x, y in train_data:
            # x=x.squeeze(0)
            X_train.append(x.numpy())
            Y_train.append(y)
        X_train = np.array(X_train)
        X_train = X_train.transpose(0, 2, 3, 1)
        X_train_rgb = []
        #self.X_231=X_train
        for img in X_train:
            img = img * std + mean
            X_train_rgb.append(img)
        '''
        for img in X_train_rgb:
            for i in range(0, 28):
                for j in range(0, 28):
                    if img[i][j].tolist() != [.5, .5, .5]:
                        img[i][j] = np.array([0, 128, 128])
        '''
        #X_train_rgb=np.array(X_train_rgb)
        #X_train_rgb = X_train_rgb.transpose(0, 3, 1, 2)
        self.images = X_train_rgb
        self.label=torch.tensor(Y_train)
        #print(self.label.shape)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        label = self.label[index]

        return image, label 


if __name__ == '__main__':
    '''
    dataset = MyDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=600, shuffle=True)
    for batch_image, batch_labels in dataloader:
        color_mnist(batch_image)
        
        plt.imshow(batch_image[0])
        plt.savefig("./minist.png")
        print('save finish')
        plt.show()
        
    '''
    print(choose_color(0))