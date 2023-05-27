import pandas as pd 
import numpy as np
import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
from change_data import *
import joblib
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
torch.cuda.set_device(2)
def add_missing_columns(d, columns) :
    missing_col = set(columns) - set(d.columns)
    for col in missing_col :
        d[col] = 0
        
def fix_columns(d, columns):  
    add_missing_columns(d, columns)
    assert(set(columns) - set(d.columns) == set())
    d = d[columns]
    return d

def data_process(df, model) :
    df.replace(" ?", pd.NaT, inplace = True)
    if model == 'train' :
        df.replace(" >50K", 1, inplace = True)
        df.replace(" <=50K", 0, inplace = True)
    if model == 'test':
        df.replace(" >50K.", 1, inplace = True)
        df.replace(" <=50K.", 0, inplace = True)
        
    trans = {'workclass' : df['workclass'].mode()[0], 'occupation' : df['occupation'].mode()[0], 'native-country' : df['native-country'].mode()[0]}
    df.fillna(trans, inplace = True)
    df.drop('fnlwgt', axis = 1, inplace = True)
    df.drop('capital-gain', axis = 1, inplace = True)
    df.drop('capital-loss', axis = 1, inplace = True)

    df_object_col = [col for col in df.columns if df[col].dtype.name == 'object']
    df_int_col = [col for col in df.columns if df[col].dtype.name != 'object' and col != 'income']
    target = df["income"]
    dataset = pd.concat([df[df_int_col], pd.get_dummies(df[df_object_col])], axis = 1)
    return target, dataset
        

class Adult_data(Dataset) :
    def __init__(self, model) :
        super(Adult_data, self).__init__()
        self.model = model
        
        #df_train = pd.read_csv('AXY/adult/adult.csv', header = None, names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
        #df_train=df_train_changed
        df_train = pd.read_csv('AXY/adult/adult.csv', header = None, names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
        df_train_np=np.array(df_train)
        pxa_list,index_a_list,index_x_list=change_sex_and_get_pxa(df_train_np,100,100)
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        df_train_changed=pd.DataFrame(df_train_np, columns=column_names)
        df_test = pd.read_csv('AXY/adult/adult.test', header = None, skiprows = 1, names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
        #print('12345',df_train[['sex']])
        train_target, train_dataset = data_process(df_train_changed, 'train')
        train_dataset.info()
        #print(1233345,train_dataset[['sex_ Male']])
        test_target, test_dataset = data_process(df_test, 'test')
        
#         进行独热编码对齐
        test_dataset = fix_columns(test_dataset, train_dataset.columns)
#         print(df["income"])
        train_dataset = train_dataset.apply(lambda x : (x - x.mean()) / x.std())
        test_dataset = test_dataset.apply(lambda x : (x - x.mean()) / x.std())
#         print(train_dataset['native-country_ Holand-Netherlands'])
        
        train_target, test_target = np.array(train_target), np.array(test_target)
        train_dataset, test_dataset = np.array(train_dataset, dtype = np.float32), np.array(test_dataset, dtype = np.float32)
        if model == 'test' :
            isnan = np.isnan(test_dataset)
            test_dataset[np.where(isnan)] = 0.0
#             print(test_dataset[ : , 75])
        
        if model == 'test':
            self.target = torch.tensor(test_target, dtype = torch.int64)
            self.dataset = torch.FloatTensor(test_dataset)
        else :
#           前百分之八十的数据作为训练集，其余作为验证集
            if model == 'train' : 
                self.target = torch.tensor(train_target, dtype = torch.int64)[ : int(len(train_dataset) * 0.8)]
                self.dataset = torch.FloatTensor(train_dataset)[ : int(len(train_target) * 0.8)]
            else :
                self.target = torch.tensor(train_target, dtype = torch.int64)[int(len(train_target) * 0.8) : ] 
                self.dataset = torch.FloatTensor(train_dataset)[int(len(train_dataset) * 0.8) : ]
        print(self.dataset.shape, self.target.dtype)   
        
    def __getitem__(self, item) :
        return self.dataset[item], self.target[item]
    
    def __len__(self) :
        return len(self.dataset)


train_dataset = Adult_data(model = 'train')
#print('.........',train_dataset[0])
test_dataset = Adult_data(model = 'test')

train_loader = DataLoader(train_dataset, batch_size = 100, shuffle = False, drop_last = False)

test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, drop_last = False)


class Adult_Model(nn.Module) :
    def __init__(self) :
        super(Adult_Model, self).__init__()
        self.net = nn.Sequential(nn.Linear(102, 64), 
                                nn.ReLU(), 
                                nn.Linear(64, 32), 
                                nn.ReLU(),
                                nn.Linear(32, 2)
                                )
    def forward(self, x) :
        out = self.net(x) 
#         print(out)
        return F.softmax(out)


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = Adult_Model().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
criterion = nn.CrossEntropyLoss()
max_epoch = 30
classes = [' <=50K', ' >50K']
mse_loss = 1000000
os.makedirs('MyModels', exist_ok = True)
writer = SummaryWriter(log_dir = 'logs')

df_train = pd.read_csv('AXY/adult/adult.csv', header = None, names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
df_train_np=np.array(df_train)
pxa_list,index_a_list,index_x_list=change_sex_and_get_pxa(df_train_np,100,100)
joblib.dump(pxa_list,'AXY/adult/adult_pro/pxa')
for epoch in range(max_epoch) :
    
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    loader_term=0
    pyax_list=[]
    for x, label in train_loader :
        pyax=pyax=np.ones([2,2,2], dtype=float)
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        
        out = model(x)
        loss = criterion(out, label)
        train_loss += loss.item()
        loss.backward()
        
        _, pred = torch.max(out, 1)
#         print(pred)
        num_correct = (pred == label).sum().item()
        acc = num_correct / x.shape[0]
        train_acc += acc
        optimizer.step()
        if epoch==30-1:
            for a_index in  index_a_list[loader_term]:
                for x_index in index_x_list[loader_term]:
                    for y_index in pred:
                        pyax[a_index,x_index,y_index.detach()]+=1
            for board in pyax:
                #ax_sum=board.sum(axis=(0,1))
                for line in board:
                    if line.sum()!=0:
                        line/=line.sum()
            pyax_list.append(pyax)
        loader_term+=1
        if loader_term==100:break
    if epoch==30-1:
        pyax_list=np.array(pyax_list)
        joblib.dump(pyax_list,'AXY/adult/adult_pro/pyax')
    print(f'epoch : {epoch + 1}, train_loss : {train_loss / len(train_loader.dataset)}, train_acc : {train_acc / len(train_loader)}')
    writer.add_scalar('train_loss', train_loss / len(train_loader.dataset), epoch)
    


best_model = Adult_Model().to(device)
#ckpt = torch.load('AXY/adult/Deeplearning_Model.pth', map_location='cpu')
#best_model.load_state_dict(ckpt)

test_loss = 0.0
test_acc = 0.0
best_model.eval()
result = []

for x, label in test_loader :
    x, label = x.to(device), label.to(device)
    
    out = best_model(x)
    loss = criterion(out, label)
    test_loss += loss.item()
    _, pred = torch.max(out, dim = 1)
    result.append(pred.detach())
    num_correct = (pred == label).sum().item()
    acc = num_correct / x.shape[0]
    test_acc += acc

print(f'test_loss : {test_loss / len(test_loader.dataset)}, test_acc : {test_acc / len(test_loader)}')


'''result = torch.cat(result, dim = 0).cpu().numpy()
with open('AXY/adult/Predict/Deeplearing.csv', 'w', newline = '') as file :
    writer = csv.writer(file)
    writer.writerow(['id', 'pred_result'])
    for i, pred in enumerate(result) :
        writer.writerow([i, classes[pred]])'''
