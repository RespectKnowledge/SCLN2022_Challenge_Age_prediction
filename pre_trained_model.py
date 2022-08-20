# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:59:06 2022

@author: Administrateur
"""

import timm
import torch.nn as nn
# model = timm.create_model('vit_base_patch16_224', pretrained=True)
# #model.patch_embed=nn.Conv2d(3, 786,kernel_size=(16, 16))
# model.head = nn.Linear(model.head.in_features, 3)

import torch
# out=model(torch.rand(1,3,200,200))

##### densnet201 model
import torchvision.models as models

model = models.densenet201(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier= nn.Linear(num_ftrs, 1)
model.features[0]=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

out=model(torch.rand(1,4,200,200))

children_counter = 0
for n,c in model.named_children():
    print("Children Counter: ",children_counter," Layer Name: ",n,)
    children_counter+=1
    
model._modules
from torchsummary import summary
summary(model,input_size=(3, 224, 224))
# l = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
# named_layers = dict(model.named_modules())

# child_counter = 0
# for child in model.children():
#    print(" child", child_counter, "is:")
#    print(child)
#    child_counter += 1


#### remodified 2D pretrained model
import torchvision.models as models
model_ft = models.resnet18(pretrained=True)
model_ft.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
## Modify fc layers to match num_classes
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,3)

out=model_ft(torch.rand(1,4,200,200))


