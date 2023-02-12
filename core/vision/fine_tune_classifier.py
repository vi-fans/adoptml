#example of use: python3 fine_tune_classifier.py train_dir/ test_dir/ model.pth
import os
import sys

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import cv2
from torchvision import datasets,models,transforms

import random
import numpy as np

if __name__=='__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    model=None
    model=models.squeezenet1_1(pretrained=True)
    model.classifier[1]=nn.Conv2d(512,len(os.listdir(sys.argv[1])),kernel_size=(1,1),stride=(1,1))
    model.to(torch.device('cuda'))

    train_transforms=transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    test_transforms=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset=datasets.ImageFolder(sys.argv[1],train_transforms)
    val_dataset=datasets.ImageFolder(sys.argv[1],test_transforms)
    test_dataset=datasets.ImageFolder(sys.argv[2],test_transforms)

    train_dataloader=DataLoader(train_dataset,batch_size=8)
    val_dataloader=DataLoader(val_dataset,batch_size=1)
    test_dataloader=DataLoader(test_dataset,batch_size=1)

    optimiser=torch.optim.SGD(model.parameters(),lr=0.001)
    criterion=torch.nn.CrossEntropyLoss()

    for epoch in range(1000):
        model.train()

        epoch_loss=0
        correct=0
        total=0
        for img,y in train_dataloader:
            img=img.to('cuda')
            y=y.to('cuda')
            optimiser.zero_grad()
            prob=model(img)
            loss=criterion(prob,y)
            _,prediction=torch.max(prob,1)
            correct+=torch.sum(y==prediction)
            total+=len(y)
            loss.backward()
            optimiser.step()
            epoch_loss+=loss.item()
        print('epoch:',epoch,'loss:',epoch_loss,'accuracy:',correct/total,'correct:',correct,'total:',total)
       
        if epoch%10==0: 
            model.eval()
            correct=0
            total=0
            with torch.no_grad():
                for img,y in val_dataloader:
                    img=img.to('cuda')
                    y=y.to('cuda')
                    prob=model(img)
                    _,prediction=torch.max(prob,1)
                    correct+=torch.sum(y==prediction)
                    total+=len(y)
                print('train accuracy:',correct/total,'correct:',correct,'total:',total)

            model.eval()
            correct=0
            total=0
            with torch.no_grad():
                for img,y in test_dataloader:
                    img=img.to('cuda')
                    y=y.to('cuda')
                    prob=model(img)
                    _,prediction=torch.max(prob,1)
                    correct+=torch.sum(y==prediction)
                    total+=len(y)
                print('test accuracy:',correct/total,'correct:',correct,'total:',total)

            model.eval()
            for eval_dir in [sys.argv[2],sys.argv[1]]:
                correct=0
                total=0
                cls_list=train_dataset.classes

                with torch.no_grad():
                    for cls in os.listdir(eval_dir):
                        for img_file in os.listdir(eval_dir+cls):
                            query_img=Image.open(eval_dir+cls+'/'+img_file).convert('RGB')
                            query_tensor=test_transforms(
                                query_img
                            )
                            query_tensor=torch.Tensor(np.array([query_tensor.numpy()]))
                            prob=model(
                                query_tensor.cuda()
                            )
                            _,found_cls_id=torch.max(prob,1)
                            print(img_file,cls_list[found_cls_id])
                            if cls_list[found_cls_id]==cls:
                                correct+=1
                            total+=1
                    print('test accuracy:',correct/total,'correct:',correct,'total:',total)

    torch.save(model.state_dict(),sys.argv[3])

