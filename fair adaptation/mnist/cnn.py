from __future__ import print_function
import joblib
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from mnist import *
import mnist
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import random
# 如果有gpu的可使用GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_set = MyDataset(train_data)
test_set = MyDataset(test_data)
train_dl = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=1)
test_dl = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=1)
import torch.nn.functional as F
 
num_classes = 10  # 图片的类别数

color_list=[[0, 0, 128],[0, 128,0],[128, 0, 0],[0, 128, 128],[128, 0, 128],
            [128, 128, 0],[128, 128, 128],[255, 0, 0],[0, 255, 0],[0, 255, 255]]
    
class Model(nn.Module):
     def __init__(self):
        super().__init__()
         # 特征提取网络
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3) 
        self.pool1 = nn.MaxPool2d(2)                
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3) 
        self.pool2 = nn.MaxPool2d(2) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) 
        self.pool3 = nn.MaxPool2d(2) 
                                      
        # 分类网络
        self.fc1 = nn.Linear(128, 64)          
        self.fc2 = nn.Linear(64, num_classes)
     # 前向传播
     def forward(self, x):
        x=x.float()
        x = self.pool1(F.relu(self.conv1(x)))     
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
 
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
       
        return x

from torchinfo import summary
# 将模型转移到GPU中（我们模型运行均在GPU中进行）
model = Model().to(device)
 
summary(model)

loss_fn    = nn.CrossEntropyLoss() # 创建损失函数
learn_rate = 1e-2 # 学习率
opt        = torch.optim.SGD(model.parameters(),lr=learn_rate)

# 训练循环
def train(dataloader, model, loss_fn, optimizer,index_a_list,index_x_list,epoch):
    pyax_list=[]
    size = len(dataloader.dataset)  # 训练集的大小，一共60000张图片
    num_batches = len(dataloader)   # 批次数目，1875（60000/32）
    
    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率
    index=0
    for X, y in dataloader:  # 获取图片及其标签
        pyax=np.ones([10,10,10], dtype=float)
        X=np.array(X)
        X = X.transpose(0, 3, 1, 2)
        X=torch.tensor(X)
        X, y = X.to(device), y.to(device)
        
        # 计算预测误差
        pred = model(X)          # 网络输出
        loss = loss_fn(pred, y)  # 计算网络输出和真实值之间的差距，targets为真实值，计算二者差值即为损失
        
        # 反向传播
        optimizer.zero_grad()  # grad属性归零
        loss.backward()        # 反向传播
        optimizer.step()       # 每一步自动更新
        
        # 记录acc与loss
        train_acc  += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()
        
        if epoch==20-1:
            for a_index in  index_a_list[index]:
                for x_index in index_x_list[index]:
                    for y_index in pred:
                        pyax[a_index,x_index,torch.argmax(y_index.detach()).item()]+=1
            for board in pyax:
                #ax_sum=board.sum(axis=(0,1))
                for line in board:
                    if line.sum()!=0:
                        line/=line.sum()
            pyax_list.append(pyax)
        index+=1      
        if index==100:
            break
    if epoch==20-1:
        pyax_list=np.array(pyax_list)
        joblib.dump(pyax_list,'program/causal-adaptation-speed-master/AXY/mnist/pro_matrix/pyax')
    train_acc  /=100*100         #size
    train_loss /=100          #num_batches
 
    return train_acc, train_loss,pyax_list

def test (dataloader, model, loss_fn):
    size        = len(dataloader.dataset)  # 测试集的大小，一共10000张图片
    num_batches = len(dataloader)          # 批次数目，313（10000/32=312.5，向上取整）
    test_loss, test_acc = 0, 0
    
    # 当不进行训练时，停止梯度更新，节省计算内存消耗
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs=np.array(imgs)
            imgs = imgs.transpose(0, 3, 1, 2)
            imgs=torch.tensor(imgs)
            imgs, target = imgs.to(device), target.to(device)
            
            # 计算loss
            target_pred = model(imgs)
            loss        = loss_fn(target_pred, target)
            
            test_loss += loss.item()
            test_acc  += (target_pred.argmax(1) == target).type(torch.float).sum().item()
 
    test_acc  /= size
    test_loss /= num_batches
 
    return test_acc, test_loss

epochs     = 20
train_loss = []
train_acc  = []
test_loss  = []
test_acc   = []


def colorp_and_get_pxa():
    index_1=0
    pxa_list=[]
    index_a_list=[]
    index_x_list=[]
    for X, y in train_dl:  # 获取图片及其标签
        color_pro,ini_color_index=mnist.color_mnist(X,index_1)
        #print(ini_color_index)
        pxa=np.ones([10,10], dtype=float)
        #print(pxa.shape)
        index_a_list_batch=[]
        for img_index,img in enumerate(X):
            color,color_index=mnist.choose_color(index_1)
            index_a_list_batch.append(color_index[0])
            if random.random()<0.5:
                for i in range(0, 28):
                    for j in range(0, 28):
                        #print(mnist.random_unit(0.5))
                        #print(ini_color_index[img_index],color_index)
                        if img[i][j].tolist() != [.5, .5, .5]:
                            img[i][j] = torch.tensor(np.array(color)).double()
                ini_color_index[img_index]=color_index
            #print(pxa[color_index,ini_color_index[img_index]])
            #print(ini_color_index[img_index])
            #print(color_index,ini_color_index[img_index])
            pxa[color_index,ini_color_index[img_index]]+=1
            #print(pxa[color_index,ini_color_index[img_index]])
        index_1=index_1+1
        for line in pxa:
            if line.sum()!=0:
                line /=line.sum()
        pxa_list.append(pxa)
        index_x_list.append(ini_color_index)
        index_a_list.append(index_a_list_batch)
        if index_1==100:
            break
    joblib.dump(mnist.pa,'program/causal-adaptation-speed-master/AXY/mnist/pro_matrix/pa')
    joblib.dump(pxa_list,'program/causal-adaptation-speed-master/AXY/mnist/pro_matrix/pxa')
    return pxa_list,index_a_list,index_x_list

pxa_list,index_a_list,index_x_list=colorp_and_get_pxa()
pxa_list,index_a_list,index_x_list=np.array(pxa_list,dtype=object),np.array(index_a_list,dtype=object),np.array(index_x_list,dtype=object)
'''print(pxa_list.shape,index_a_list.shape,index_x_list.shape)
_,_,pyax_list=train(train_dl, model, loss_fn, opt,index_a_list,index_x_list)
print(pyax_list.shape)'''

for epoch in range(epochs):

    model.train()
    epoch_train_acc, epoch_train_loss,pyax_list = train(train_dl, model, loss_fn, opt,index_a_list,index_x_list,epoch)
    
    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_fn)
    time.sleep(0.003)
    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)
    
    template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%，Test_loss:{:.3f}')
    print(template.format(epoch+1, epoch_train_acc*100, epoch_train_loss, epoch_test_acc*100, epoch_test_loss))
print('Done')