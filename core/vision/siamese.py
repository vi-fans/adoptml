#example of use: python3 siamese.py train_dir/ test_dir/ model.pth
import os
import sys

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader

import cv2
from torchvision import models,transforms

import random
import numpy as np

class siamese_dataset(Dataset):
    def __init__(self,path):
        super(siamese_dataset,self).__init__()
        self.data,self.num_class,self.c=self.load(path)
        self.transforms=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def load(self,path):
        data={}
        idx=0
        c=0
        for cls in os.listdir(path):
            data[idx]=[]
            for img_file in os.listdir(path+'/'+cls):
                img=Image.open(path+'/'+cls+'/'+img_file).convert('RGB')
                data[idx].append(img)
                c+=1
            idx+=1
        return data,idx,c
    def __len__(self):
        return self.c
    def __getitem__(self,index):
        label=None
        first_img=None
        second_img=None
        if index%2==1:
            label=0.0
            idx=random.randint(0,self.num_class-1)
            first_img_idx=random.randint(0,len(self.data[idx])-1)
            second_img_idx=random.randint(0,len(self.data[idx])-1)
            while first_img_idx==second_img_idx:
                second_img_idx=random.randint(0,len(self.data[idx])-1)
            first_img=self.data[idx][first_img_idx]
            second_img=self.data[idx][second_img_idx]
        else:
            label=1.0
            first_idx=random.randint(0,self.num_class-1)
            second_idx=random.randint(0,self.num_class-1)
            while first_idx==second_idx:
                second_idx=random.randint(0,self.num_class-1)
            first_img=random.choice(self.data[first_idx])
            second_img=random.choice(self.data[second_idx])
        label=torch.from_numpy(np.array([label],dtype=np.float32))
        first_img=self.transforms(first_img)
        second_img=self.transforms(second_img)
        return first_img,second_img,label

class siamese(nn.Module):
    def __init__(self):
        super(siamese,self).__init__()
        self.features=models.vgg11_bn(pretrained=True).features
        for param in self.features.parameters():
            param.requires_grad=True
        self.fc=nn.Sequential(
            nn.Linear(25088,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,32),
        )
        for param in self.fc.parameters():
            param.requires_grad=True
    def forward(self,img_query,img_reference):
        features_query=self.features(img_query)
        features_reference=self.features(img_reference)
        features_query=self.fc(
            features_query.view(features_query.size()[0],-1)
        )
        features_reference=self.fc(
            features_reference.view(features_reference.size()[0],-1)
        )
        return features_query,features_reference

class contrastive_loss(nn.Module):
    def __init__(self):
        super(contrastive_loss,self).__init__()
    def forward(self,first_output,second_output,label):
        d=F.pairwise_distance(first_output,second_output,keepdim=True)
        loss=torch.mean(
            (1-label)*torch.pow(d,2)+
            label*torch.pow(
                torch.clamp(2-d,min=0.0),2
            )
        )
        return loss

if __name__=='__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    model=siamese()
    model.to(torch.device('cuda'))

    train_dataset=siamese_dataset(sys.argv[1])
    train_dataloader=DataLoader(train_dataset,batch_size=4,shuffle=True)

    test_dataset=siamese_dataset(sys.argv[2])
    test_dataloader=DataLoader(test_dataset,batch_size=4)

    optimiser=torch.optim.Adam(model.parameters(),lr=0.001)
    criterion=contrastive_loss()

    for epoch in range(100):
        model.train()

        epoch_loss=[]
        for batch_id,(first_img,second_img,y) in enumerate(train_dataloader,0):
            first_img,second_img,y=first_img.cuda(),second_img.cuda(),y.cuda()
            first_img,second_img,y=Variable(first_img),Variable(second_img),Variable(y)
            optimiser.zero_grad()
            first_vec,second_vec=model(first_img,second_img)
            loss=criterion(first_vec,second_vec,y)
            epoch_loss.append(loss.item())
            loss.backward()
            optimiser.step()
        epoch_loss=np.array(epoch_loss)
        print('epoch:',epoch,'loss:',np.average(epoch_loss))

    torch.save(model.state_dict(),sys.argv[3])

