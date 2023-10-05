# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 02:04:27 2023

@author: Lab639
"""

import torch
import numpy as np
from dataloader import RetinopathyLoader
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
from dataloader import RetinopathyLoader
import seaborn as sn
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 5
momentum = 0.9
weight_decay = 5e-4
lr = 0.0015

def train(model, train_loader, test_loader, optimizer, device, num_class, name_type):
    df = pd.DataFrame()
    df['epoch'] = range(1, epochs+1)
    best_model_weights=None
    best_evaluated_acc=0
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    list_acc_train = []
    list_acc_test = []
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        correct = 0
        for _, (images, label) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            predict = model(images)
            loss = criterion(predict, label)
            total_loss += loss.item()
            pred = predict.argmax(dim=1)
            for i in range(len(label)):
                if pred[i] == label[i]:
                    correct +=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss = total_loss / len(train_loader.dataset)
        acc_train = 100. * correct / len(train_loader.dataset)
        list_acc_train.append(acc_train)
        print(f'epoch{epoch:>2d} loss:{total_loss:.4f} acc_train:{acc_train:.3f}%')
        
        with torch.no_grad():
            acc_test,_ = evaulate(model, test_loader, device, num_class)
            list_acc_test.append(acc_test)
            print(f'epoch{epoch:>2d}  acc_test:{acc_test:.3f}%')
            if acc_test>best_evaluated_acc:
                best_evaluated_acc = acc_test
                best_model_weights = copy.deepcopy(model.state_dict())
        torch.cuda.empty_cache()  
        
    df['acc_train'] = list_acc_train
    df['acc_test'] = list_acc_test
    print("best acc ", name_type, ": ",best_evaluated_acc)

    # save model
    torch.save(best_model_weights, os.path.join(name_type+'.pth'))
    model.load_state_dict(best_model_weights)
    return df

def evaulate(model, test_loader, device, num_class):
    confusion_matrix=np.zeros((num_class,num_class))
    model.eval()
    correct = 0
    for _, (images, label) in enumerate(test_loader):
        images = images.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        predict = model(images)
        pred = predict.argmax(dim=1)
        for i in range(len(label)):
            confusion_matrix[int(label[i])][int(pred[i])]+=1
            if pred[i] == label[i]:
                correct +=1
    acc = 100. * correct / len(test_loader.dataset)
    #normalize
    confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1).reshape(num_class,1)
    return acc, confusion_matrix

def plot(df1, df2, title):
    # fig = plt.figure(figsize=(10,6))
    plt.title(title, fontsize=18)
    plt.ylabel("Accuracy(%)", fontsize=18)
    plt.xlabel("Epoch", fontsize=18)
    for name in df1.columns[1:]:
        plt.plot(range(1,1+len(df1)), name, data=df1, label=name[4:]+'(w/o pretraining)')
    for name in df2.columns[1:]:
        plt.plot(range(1,1+len(df2)), name, data=df2, label=name[4:]+'(with pretraining)')
    plt.legend()
    plt.show()

def plot_confusion_matrix(confusion_matrix):
    df_cm = pd.dataframe(confusion_matrix, range(5), range(5))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    plt.show()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.xaxis.set_label_position('bottom')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.show()
    return fig

if __name__== "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size_18 = 50
    batch_size_50 = 8
    num_class= 5
    
    train_dataset = RetinopathyLoader('gamma_train', mode="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size_50, shuffle=True, num_workers=4, drop_last=True)
    test_dataset = RetinopathyLoader('gamma_test', mode="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size_50, shuffle=True, num_workers=4, drop_last=True)
    
    # resnet18 with pretrained
    print('resnet18 with pretrained')
    model_with_18 = models.resnet18(pretrained=True)
    optimizer=optim.SGD(model_with_18.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    df_with_pretrain_18 = train(model_with_18, train_loader, test_loader, optimizer=optimizer, device=device, num_class=num_class, name_type='resnet18_with_pretrain')
    _, confusion_matrix = evaulate(model_with_18, test_loader, device, num_class)
    fig = plot_confusion_matrix(confusion_matrix)
    fig.savefig('ResNet18_with_pretrained_weights.png')
    
    # resnet50 with pretrained
    # print('resnet50 with pretrained')
    # model_with_50 = models.resnet50(pretrained=True)    
    # optimizer=optim.SGD(model_with_50.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    # df_with_pretrain_50 = train(model_with_50, train_loader, test_loader, optimizer=optimizer, device=device, num_class=num_class, name_type='resnet50_with_pretrain')
    # _, confusion_matrix = evaulate(model_with_50, test_loader, device, num_class)
    # print(confusion_matrix)
    # fig = plot_confusion_matrix(confusion_matrix)
    # fig.savefig('ResNet50_with_pretrained_weights.png')
    
    # # resnet18 without pretrained
    # print('resnet18 wo pretrained')
    # model_wo_18 = models.resnet18(pretrained=False)
    # optimizer=optim.SGD(model_wo_18.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    # df_wo_pretrain_18 = train(model_wo_18, train_loader, test_loader, optimizer=optimizer, device=device, num_class=num_class, name_type='resnet18_wo_pretrain')
    # _, confusion_matrix = evaulate(model_wo_18, test_loader, device, num_class)
    # print(confusion_matrix)
    # fig = plot_confusion_matrix(confusion_matrix)
    # fig.savefig('ResNet18_wo_pretrained_weights.png')
    
    # # resnet50 without pretrained
    # print('resnet50 wo pretrained')
    # model_wo_50 = models.resnet50(pretrained=False)
    # optimizer=optim.SGD(model_wo_50.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    # df_wo_pretrain_50 = train(model_wo_50, train_loader, test_loader, optimizer=optimizer, device=device, num_class=num_class, name_type='resnet50_wo_pretrain')
    # _, confusion_matrix = evaulate(model_wo_50, test_loader, device, num_class)
    # print(confusion_matrix)
    # fig = plot_confusion_matrix(confusion_matrix)
    # fig.savefig('ResNet50_wo_pretrained_weights.png')
    
   