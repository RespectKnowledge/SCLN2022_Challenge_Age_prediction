# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:06:26 2022

@author: Administrateur
"""
##### SCLN challenege age predicition
import numpy as np 
import nibabel as nb
import os
import torch
from torch.utils.data import Dataset
class Age_birth(Dataset):
    def __init__(self,root,datapath,pattern=None):
        self.root=root
        self.datapath=datapath
        self.pattern=pattern
        self.train_f=np.load(self.datapath,allow_pickle=True)
        self.dataleng=self.train_f.shape[0]
    def __getitem__(self, index):
        
        ###### extract L shape pattern
        paths=self.train_f[index][0]+'_L.shape.gii' ## subject path
        pma_age= self.train_f[index][1] ## pma age
        GA_age= self.train_f[index][2] ## GA age
        #print(paths)
        #print(pma_age)
        #print(GA_age)
        subj_path=os.path.join(self.root,paths)
        #print(subj_path)
        image = nb.load(subj_path)
        img_array=self.extract_array1(image)
        img_array = np.swapaxes(img_array, 2,0) 
        
        ####this is pytorch normalization function
        #img_t=torch.nn.functional.normalize(torch.from_numpy(img_array).float()) #normalize tensor
        
        ######### normalize the np array between zero and 1
        img_array = (img_array - np.min(img_array))/np.ptp(img_array)
        
        ##### convert into pytorch tensor
        img_t=torch.from_numpy(img_array).float() #normalize tensor
        #print(img_array.shape)
        ###### convert age into float tensor from np array
        GA_age=torch.as_tensor(np.array(GA_age).astype('float'))
        pma_age=torch.as_tensor(np.array(pma_age).astype('float'))
        #### return features,age at birth and age at scan
        return img_t,GA_age,pma_age
    
    #### extract features from image object with channels
    def extract_array(self,image):
        #### Define number of channels in gifti ####
        channels=4 ### Each channel represents a structural cortical metric (myelin, curvature, cortical thickness and sulcal depth - in that order) 

        img_array=[]


        #### Iterate over each channel, extract array and append to img_array object/list ####  
        for i in range(channels):
            array = image.darrays[i].data
            img_array.append(array)
        #### Swap axes so that img_array dim is 40962,4 - this will depend on how participants want to work with the image arrays #### 
        img_array = np.swapaxes(img_array, 0,1) 
        return img_array
            
    def extract_array1(self,image):
        #### Define number of channels in gifti ####
        channels=4 ### Each channel represents a structural cortical metric (myelin, curvature, cortical thickness and sulcal depth - in that order) 

        img_array=[]


        #### Iterate over each channel, extract array and append to img_array object/list ####  
        for i in range(channels):
            array = image.darrays[i].data
            array=array[0:40000]
            array1=array.reshape([200, 200])
            img_array.append(array1)
        
        #### Swap axes so that img_array dim is 40962,4 - this will depend on how participants want to work with the image arrays #### 
        img_array = np.swapaxes(img_array,0,2) 
        return img_array
        
        # #### Swap axes so that img_array dim is 40962,4 - this will depend on how participants want to work with the image arrays #### 
        # img_array = np.swapaxes(img_array, 0,1) 
        # return img_array
    
    def __len__(self):
        
        return (self.dataleng)
 
pathn='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\regression_native_space_features\\regression_native_space_features'
#pathtemplate='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\regression_template_space_features\\regression_template_space_features'
pathtrain='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\Regression\\birth_age_confounded\\train.npy' 

pathvalid='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\Regression\\birth_age_confounded\\validation.npy'     
    
train_set=Age_birth(pathn,pathtrain) 
#datatrain=Age_birth(pathtemplate,pathtrain)

# im,age1,age2=datatrain[0]
# for i in range(0,len(datatrain)):
#     im,age1,age2=datatrain[i]
#     print(im.shape)
#     print(im.min())
#     print(im.max())
#     #print(age1)
#     #print(age2)
    
 
test_set=Age_birth(pathn,pathvalid) 
#datavalid=Age_birth(pathtemplate,pathvalid) 
    
from torch.utils.data import DataLoader
train_loader=DataLoader(train_set,batch_size=4,shuffle=True,pin_memory=True)

valid_loader=DataLoader(test_set,batch_size=4,shuffle=False,pin_memory=True)

for r,d in enumerate(valid_loader):
    data=d
    im_b,age_b,age2_b=data
    print(im_b.shape)
    print(age_b)
    break

#%%

## model

import torch
import torch.nn as nn
#Layer1 40962-3+1=40960/2=Layer2 20480-5+1=20476/2Layer 3 10238-11+1=10228/2=
#Layer 4 5114-3+1=5112=2556Layer5 2556-5+1=2552/2=Layer6 1276-5+1=1272/2=Layer7 636-5+1=632Layer 8 632/2=316-5+1=
#312/2=156-5+1=154/2=76
class CNN_model(nn.Module):
  def __init__(self,in_chan,classes):
    super(CNN_model,self).__init__()
    # block 1    40962x4
    self.c11=nn.Conv1d(in_channels=in_chan,
                       out_channels=256,kernel_size=3) # 40962-3+1=40960
    self.maxpool11=nn.MaxPool1d(2) #20480
    self.c12=nn.Conv1d(in_channels=256,
                       out_channels=128,kernel_size=5) #20480-5+1=20476
    self.maxpool12=nn.MaxPool1d(2) #10238
    # block2
    self.c21=nn.Conv1d(in_channels=128,
                       out_channels=128,kernel_size=11) #10238-11+1=10228
    self.maxpool21=nn.MaxPool1d(2) #10228/2=5114
    self.c22=nn.Conv1d(in_channels=128,
                       out_channels=128,kernel_size=3) #5114-3+1=5112 
    self.maxpool22=nn.MaxPool1d(2) #5112/2=2556

    # block3
    self.c31=nn.Conv1d(in_channels=128,
                       out_channels=128,kernel_size=5) #2556-5+1=2552
    self.maxpool31=nn.MaxPool1d(2) #2552/2=1276
    self.c32=nn.Conv1d(in_channels=128,
                       out_channels=128,kernel_size=5) #1276-5+1=1272 
    self.maxpool32=nn.MaxPool1d(2) #1272/2=636

    # block4
    self.c41=nn.Conv1d(in_channels=128,
                       out_channels=64,kernel_size=5) #636-5+1=632
    self.maxpool41=nn.MaxPool1d(2) #632/2=316
    self.c42=nn.Conv1d(in_channels=64,
                       out_channels=64,kernel_size=5) #316-5+1=312 
    self.maxpool42=nn.MaxPool1d(2) #312/2=156

    # block 5
    self.c51=nn.Conv1d(in_channels=64,
                       out_channels=64,kernel_size=5) #156-5+1=152
    self.maxpool51=nn.MaxPool1d(2) #152/2=76
    
    
    # linear layer
    self.fc1=nn.Linear(64*76,128)
    self.fc2=nn.Linear(128,classes)

  def forward(self,x):
    # block 1 
    x=self.maxpool12(self.c12(self.maxpool11(self.c11(x))))
    # block 2
    x=self.maxpool22(self.c22(self.maxpool21(self.c21(x))))
    
    # block 3
    x=self.maxpool32(self.c32(self.maxpool31(self.c31(x))))

    # block 4 
    x=self.maxpool42(self.c42(self.maxpool41(self.c41(x))))
    x=self.maxpool51(self.c51(x))

    #x=(self.c22(self.maxpool12(self.c12(x))))
    x=x.view(-1,64*76)
    x=self.fc1(x)
    x=self.fc2(x)
    return x
#model=CNN_model(in_chan=4,classes=1) # inp,number_layers,hidden_dim#
#inp=torch.rand(1,4,40962)
#out=model(inp)
#print(out.shape)

import torch
# out=model(torch.rand(1,3,200,200))

##### densnet201 model
import torchvision.models as models

model = models.densenet201(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier= nn.Linear(num_ftrs, 1)
model.features[0]=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#out=model(torch.rand(1,4,200,200))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import cv2
import torch
from tqdm import tqdm
"""
Loss Functions
""" 
    
def dy_huber_loss(inputs, targets, beta):
    """
    Dynamic Huber loss function
    
    """
    n = torch.abs(inputs - targets)
    cond = n <= beta
    loss = torch.where(cond, 0.5 * n ** 2, beta*n - 0.5 * beta**2)

    return loss.mean()

def dy_smooth_l1_loss(inputs, targets, beta):
    """
    Dynamic ParamSmoothL1 loss function
    
    """
    n = torch.abs(inputs - targets)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2, n + 0.5 * beta**2 - beta)

    return loss.mean()

def dy_tukey_loss(input, target, c):
    """
    Dynamic Tukey loss function
    
    """    
        
    n = torch.abs(input - target)
    cond = n <= c
    loss = torch.where(cond, ((c** 2)/6) * (1- (1 - (n /c)**2) **3 )  , torch.tensor((c** 2)/6).to('cuda'))

    return loss.mean()


"""
Evaluation Calculations
"""
models_save_path='C:\\Users\\Usuario\\Desktop\\Data\\SCLN2022\\models'

def MAE_distance(preds, labels):
    return torch.sum(torch.abs(preds - labels))

def PC_mine(preds, labels):
    dem = np.sum((preds - np.mean(preds))*(labels - np.mean(labels)))
    mina = (np.sqrt(np.sum((preds - np.mean(preds))**2)))*(np.sqrt(np.sum((labels - np.mean(labels))**2)))
    return dem/mina 


criterion = dy_smooth_l1_loss  


sigma_max = 0.7
sigma_min = 0.3
train_MAE = []
train_RMSE = []
train_PC = []
test_MAE = []    
test_RMSE = []
test_PC = []
epoch_count = []
pc_best = -2


labels2_tr = []
labels_pred_tr = []
total_loss=0

Nepochs=50
sigma_min=0.2
sigma_max=0.7
total_loss_val=0
import torch.optim as optim

for epoch in range(Nepochs):
  lr = 0.0001
  if epoch>19:
    lr = 0.0001 * 0.1  
  if epoch>29:
    lr = 0.0001 * 0.01       
  epoch_count.append(epoch)
  optimizer = optim.Adam(model.parameters(), lr =lr)
  labels2_tr = []
  labels_pred_tr = []
  total_loss=0
  total_loss_val = 0

  sigma = sigma_min + (1/2)* (sigma_max - sigma_min ) * (1+ np.cos (np.pi * ((epoch+1)/Nepochs)))
  ##### training loop
  for batch in tqdm(train_loader):
    images, labels= batch
    images = images.float().to(device)
    labels = labels.to(device)
    torch.set_grad_enabled(True)
    model.train()
    model.to(device)
    preds = model(images)
    loss = criterion(preds.squeeze(1), labels,sigma)            
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()         
    total_loss += loss.item()
    #print(labels)            
    labels2_tr.extend(labels.cpu().numpy())
    labels_pred_tr.extend(preds.detach().cpu().numpy())
    del images; del labels 
    #sigma = sigma_min + (1/2)* (sigma_max - sigma_min ) * (1+ np.cos (np.pi * ((epoch+1)/Nepochs)))
  
  labels2_ts = []
  labels_pred_ts = []
  ##### validation loop
  for batch in tqdm(valid_loader):
    images, labels= batch
    images = images.float().to(device)
    labels = labels.to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
      preds = model(images)
      total_loss_val += MAE_distance(preds.squeeze(1), labels)
      labels2_ts.extend(labels.cpu().numpy())
      labels_pred_ts.extend(preds.detach().cpu().numpy())
      del images; del labels


  ###### compute statics for testing model
  labels_pred_ts = np.squeeze(np.array(labels_pred_ts))
  labels2_ts = np.array(labels2_ts)
  labels_pred_tr = np.squeeze(np.array(labels_pred_tr))
  labels2_tr = np.array(labels2_tr)        
        
  test_MAE.append(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
  test_RMSE.append(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
  test_PC.append(float(PC_mine(labels_pred_ts, labels2_ts)))
  #test_epsilon.append(float(np.mean(1-np.exp(-((labels_pred_ts - labels2_ts)**2)/(2*(epsi_ts)**2)))))
  train_MAE.append(float(np.mean(np.abs(labels_pred_tr - labels2_tr))))
  train_RMSE.append(float(np.sqrt(np.mean((labels_pred_tr - labels2_tr)**2))))
  train_PC.append(float(PC_mine(labels_pred_tr, labels2_tr)))
  #train_epsilon.append(float(np.mean(1-np.exp(-((labels_pred_tr - labels2_tr)**2)/(2*(epsi_tr)**2)))))
  print('Ep: ', epoch, 'PC_tr: ', PC_mine(labels_pred_tr, labels2_tr), 'PC_ts: ',  PC_mine(labels_pred_ts, labels2_ts),'MAE_tr: ', total_loss/len(train_set), 'MAE_ts: ', total_loss_val/len(test_set), 'loss_tr:', total_loss/len(train_set),'loss_ts:', total_loss_val/len(test_set))
  #### check best values
  pc_best2 = float(PC_mine(labels_pred_ts, labels2_ts))
  if pc_best2 > pc_best:
    pc_best = pc_best2
    mae_best = float(np.mean(np.abs(labels_pred_ts - labels2_ts)))
    rmse_best = float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2)))
    #epsilon_best = float(np.mean(1-np.exp(-((labels_pred_ts - labels2_ts)**2)/(2*(epsi_ts)**2))))           

  #### print best values         
  print(pc_best) 
  print(mae_best) 
  print(rmse_best)   
    
  ##### save model for prediction or inferences
  model_name = 'regr1' + '_'
  #fold_name = str(opt.Fold) + '_'  
  lossfn_name = 'hyber1'
  model_name = models_save_path + model_name + lossfn_name +'.pth'  
  torch.save(model.state_dict(), model_name)      
  #train_iou.append(ious)