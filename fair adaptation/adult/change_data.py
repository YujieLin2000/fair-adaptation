import torch
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random
import pandas as pd
import numpy as np
import os
import joblib
df_train = pd.read_csv('AXY/adult/adult.csv', header = None, names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
df_train_np=np.array(df_train)
print(df_train_np[:,9])
print(df_train_np.shape)
#pa = np.random.dirichlet(np.ones(2), size=100)
#joblib.dump(pa,'AXY/adult/adult_pro/pa')
pa=joblib.load('AXY/adult/adult_pro/pa')
print(pa.shape)
def choose_sex(index):
    sex=[' Male',' Female']
    sex_index=[0,1]
    #print(pa[index])
    sex_index=np.random.choice(sex_index,100,p=pa[index])
    #print(color_index)
    sex_selected=[]
    for index in sex_index:
        sex_selected.append(sex[index])
    return sex_selected,sex_index

def change_sex_and_get_pxa(df_train_np=df_train_np,batch_size=100,n=100):
    pxa_list=[]
    index_x_list=[]
    index_a_list=[]
    for batch_num in range(n):
        index_x=[]
        pxa=np.ones([2,2], dtype=float)
        sex_selected,sex_index=choose_sex(batch_num)
        for item_num in range(batch_size):           
            if random.random()<0.5:
                df_train_np[:,9][batch_num*batch_size+item_num]=sex_selected[item_num]
            if df_train_np[:,9][batch_num*batch_size+item_num]==' Male':
                location=0
            else : location=1
            index_x.append(location)
            pxa[sex_index[batch_num],location]+=1
        for line in pxa:
                if line.sum()!=0:
                    line /=line.sum()
        pxa_list.append(pxa)
        index_a_list.append(sex_index)
        index_x_list.append(index_x)
    return pxa_list,index_a_list,index_x_list
#print(choose_sex(75))
a,b,c=change_sex_and_get_pxa(df_train_np)
a,b,c=np.array(a),np.array(b),np.array(c)
print(a.shape)
print(b.shape)
print(c.shape)