# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:27:23 2022

@author: Administrateur
"""
############   dataset description files for ############################
# The labels for these data are included in the *.npy 
# (*= train, validation and test) 
# files for the train-validation-test splits 
# (geometric-deep-learning-benchmarking/Train_Val_Test_Splits). 
# For predicting PMA at scan, these files contain two columns:
# ###### 1,2
# Subject-ID_session-ID
# PMA at scan (weeks)
# ##### 1,2,3
# For predicting GA at birth, these files contain 
#three columns:

# Subject-ID_session-ID
# PMA at scan (weeks) - this is included as an additional covariate, 
#as the prediction of GA at birth is confounded by PMA at scan.
# GA at birth (weeks)

#%%


import numpy as np
import SimpleITK as sitk
import numpy as np 
import nibabel as nb
import os

def extract_array(image):
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


pathn='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\regression_native_space_features\\regression_native_space_features'
train_f=np.load('C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\geometric-deep-learning-benchmarking-master\\geometric-deep-learning-benchmarking\\Train_Val_Test_Splits\\Regression\\birth_age_confounded\\train.npy',allow_pickle=True)
train_f.shape[0] ## 411x3(s,PMA at scan,GA at birth)
#print(train_f[0][0])
for i in range(0,train_f.shape[0]):
    ###### L shapes
    paths=train_f[i][0]+'_L.shape.gii' ## subject path
    pma_age= train_f[i][1] ## pma age
    GA_age= train_f[i][2] ## GA age
    #print(paths)
    #print(pma_age)
    #print(GA_age)
    subj_path=os.path.join(pathn,paths)
    #print(subj_path)
    image = nb.load(subj_path)
    array1 = image.darrays[0].data
    array2 = image.darrays[0].data
    array3 = image.darrays[0].data
    array4 = image.darrays[0].data
    print(array1.shape)
    print(array2.shape)
    print(array3.shape)
    print(array4.shape)
    
    img_array=extract_array(image)
    print(img_array.shape)
    break
    
    ####### R shape
    paths=train_f[i][0]+'_R.shape.gii' ## subject path
    pma_age= train_f[i][1] ## pma age
    GA_age= train_f[i][2] ## GA age
    #print(paths)
    #print(pma_age)
    #print(GA_age)
    subj_path=os.path.join(pathn,paths)
    #print(subj_path)
    image = nb.load(subj_path)
    img_array=extract_array(image)
    print(img_array.shape)
    break
#img_array1=img_array[0:40000,1]

#a_4_6 = img_array1.reshape([200, 200])
#### Load gifti image using nibabel ####
#filename = '' ### Participant to modify
# image = nb.load(filename)
# array = image.darrays[0].data
# array1 = image.darrays[1].data
# array2 = image.darrays[2].data
# array3=image.darrays[3].data
#%%
import numpy as np
import SimpleITK as sitk
import numpy as np 
import nibabel as nb
import os

def extract_array(image):
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


pathn='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\regression_native_space_features\\regression_native_space_features'
train_f=np.load('C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\Regression\\scan_age\\train.npy',allow_pickle=True)
train_f.shape[0] ## 411x3(s,PMA at scan,GA at birth)
#print(train_f[0][0])
for i in range(0,train_f.shape[0]):
    ###### L shapes
    paths=train_f[i][0]+'_L.shape.gii' ## subject path
    pma_age= train_f[i][1] ## pma age
    GA_age= train_f[i][2] ## GA age
    #print(paths)
    #print(pma_age)
    #print(GA_age)
    subj_path=os.path.join(pathn,paths)
    #print(subj_path)
    
    image = nb.load(subj_path)
    array = image.darrays.data
    img_array=extract_array(image)
    print(img_array.shape)
    break
    
    ####### R shape
    paths=train_f[i][0]+'_R.shape.gii' ## subject path
    pma_age= train_f[i][1] ## pma age
    GA_age= train_f[i][2] ## GA age
    #print(paths)
    #print(pma_age)
    #print(GA_age)
    subj_path=os.path.join(pathn,paths)
    #print(subj_path)
    image = nb.load(subj_path)
    img_array=extract_array(image)
    print(img_array.shape)
    #break
#%% dataloader
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
        paths=self.train_f[i][0]+'_L.shape.gii' ## subject path
        pma_age= self.train_f[i][1] ## pma age
        GA_age= self.train_f[i][2] ## GA age
        #print(paths)
        #print(pma_age)
        #print(GA_age)
        subj_path=os.path.join(self.root,paths)
        #print(subj_path)
        image = nb.load(subj_path)
        img_array=self.extract_array(image)
        img_array = np.swapaxes(img_array, 1,0) 
        
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
    
    def __len__(self):
        
        return (self.dataleng)
 
pathn='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\regression_native_space_features\\regression_native_space_features'
#pathtemplate='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\regression_template_space_features\\regression_template_space_features'
pathtrain='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\Regression\\birth_age_confounded\\train.npy' 

pathvalid='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\SCLN2022\\Regression\\birth_age_confounded\\validation.npy'     
    
datatrain=Age_birth(pathn,pathtrain) 
#datatrain=Age_birth(pathtemplate,pathtrain)

# im,age1,age2=datatrain[0]
# for i in range(0,len(datatrain)):
#     im,age1,age2=datatrain[i]
#     print(im.shape)
#     print(im.min())
#     print(im.max())
#     #print(age1)
#     #print(age2)
    
 
datavalid=Age_birth(pathn,pathvalid) 
#datavalid=Age_birth(pathtemplate,pathvalid) 
    
from torch.utils.data import DataLoader
train_loader=DataLoader(pathtrain,batch_size=4,shuffle=True,pin_memory=True)

valid_loader=DataLoader(datavalid,batch_size=4,shuffle=False,pin_memory=True)

for i,d in enumerate(valid_loader):
    data=d
    im_b,age_b,age2_b=data
    print(im_b.shape)
    print(age_b)
#%% model for age prediction
import torch
import torch.nn as nn



class MLP(nn.Module):

    '''
    This class defines a multi-layer perceptrons architecture
    '''

    def __init__(self,input_size,
                      hidden_size = [1024,512,32], 
                      dropout=0.5,
                      bias=False):

        super(MLP,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_linear = len(hidden_size)
        self.bias = bias

        ## fully connected layers
        self.fc_layers = torch.nn.ModuleDict()

        self.fc_layers['fc_layer_0'] = nn.Linear(self.input_size, self.hidden_size[0], bias=self.bias)
        for i in range(self.num_linear-1):
            self.fc_layers[f'fc_layer_{i+1}'] = nn.Linear(self.hidden_size[i],self.hidden_size[i+1], bias=self.bias)
        
        ## classification layer
        self.fc_class = nn.Linear(self.hidden_size[-1],1 , bias=self.bias)

        if self.dropout:
            self.dropout = nn.Dropout(self.dropout)

        #init_weights(self)

        ## activation
        self.relu = nn.ReLU()

    def forward(self,x):

        out = x
        for i in range(self.num_linear):
            out = self.fc_layers[f'fc_layer_{i}'](out)
            out = self.relu(out)
        
        if self.dropout:
            out = self.dropout(out)
        out = self.fc_class(out)
        return out
   
model=MLP(40962)
out=model(torch.rand(1,40962))
from typing import Optional, Sequence
import torch.nn as nn
class ClassifierMLP(nn.Module):
    def __init__(self, in_ch: int, num_classes: Optional[int], layers_description: Sequence[int]=(256,128), dropout_rate: float = 0.1):
        super().__init__()
        layer_list = []
        layer_list.append(nn.Linear(in_ch, layers_description[0]))
        layer_list.append(nn.ReLU())
        if dropout_rate is not None and dropout_rate > 0:
            layer_list.append(nn.Dropout(p=dropout_rate))
        last_layer_size = layers_description[0]
        for curr_layer_size in layers_description[1:]:
            layer_list.append(nn.Linear(last_layer_size, curr_layer_size))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size
        
        if num_classes is not None:
            layer_list.append(nn.Linear(last_layer_size, num_classes))
        
        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x
  
model=ClassifierMLP(40962,2)  
  
#out=model(torch.rand(1,4,40962))


#%% conversion dataset
import SimpleITK as sitk
import numpy as np 
import nibabel as nb


#### Load gifti image using nibabel ####
filename = '' ### Participant to modify
image = nb.load(filename)

#### Define number of channels in gifti ####
channels=4 ### Each channel represents a structural cortical metric (myelin, curvature, cortical thickness and sulcal depth - in that order) 

img_array=[]


#### Iterate over each channel, extract array and append to img_array object/list ####  
for i in range(channels):
    array = image.darrays[i].data
    img_array.append(array)


#### Swap axes so that img_array dim is 40962,4 - this will depend on how participants want to work with the image arrays #### 
img_array = np.swapaxes(img_array, 0,1) 


#### Create .mha image using sitk ####
mha_img = sitk.GetImageFromArray(img_array)

new_filename='' ### Participant to modify

#### Write out .mha image ####
sitk.WriteImage(mha_img, new_filename)

#### Read in .mha image again ####
mha_img1 = sitk.ReadImage(new_filename)


#### Get array from image using sitk ####
img1_array = sitk.GetArrayFromImage(mha_img1)


