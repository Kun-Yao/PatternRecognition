import pandas as pd
import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
alphabets2index = {alphabet:i for i, alphabet in enumerate(alphabets)}

TRAIN_PATH = "../dataset/train"
TEST_PATH = "../dataset/test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_onehot(text, words):
    onehot = np.zeros(62*words)
    # assert onehot.shape[0] == 124
    for i in range(len(text)):
        onehot[alphabets2index[text[i]] + i*62] = 1
    return onehot

def load_csv():
    data = pd.read_csv(f'{TRAIN_PATH}/annotations.csv').to_numpy()
    for row in data:
        if random.random() < 0.7:
            train_data.append(row)
        else:
            eval_data.append(row)
            

class TaskDataset(Dataset):
    def __init__(self, data, root, return_filename=False, task=1):
        self.data = [sample for sample in data if sample[0].startswith(f'task{task}')]
        # self.data = [sample for sample in data]
        self.return_filename = return_filename
        self.root = root
        self.task = task
    def __getitem__(self, index):
        filename, label = self.data[index]
        label = str(label)
        img = cv2.imread(f"{self.root}/{filename}")
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, (7,7))
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, (7,7))
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, (9,9))
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, (7,7))
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, (9,9))
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, (9,9))
        img = img.transpose(2, 0, 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = np.resize(img, (96, 96, 1))

        # img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        
        if self.task == 1:
            words = 1
        elif self.task == 2:
            words = 2
        elif self.task == 3:
            words = 4
        
        if self.return_filename:
            return torch.FloatTensor(img), filename
        else:
            return torch.FloatTensor(img), torch.tensor(to_onehot(label, words))

    def __len__(self):
        return len(self.data) 
   
def train(task):
    # max_valid_acc=global_acc[task]
    max_valid_acc=0
    
    if task == 1:
        train_ds = TaskDataset(train_data, root=TRAIN_PATH, task=1)
        train_dl = DataLoader(train_ds, batch_size=50, num_workers=4, drop_last=True, shuffle=True)
        eval_ds = TaskDataset(eval_data, root=TRAIN_PATH, task=1)
        eval_dl = DataLoader(eval_ds, batch_size=50, num_workers=4, drop_last=True, shuffle=True)
    elif task == 2:
        train_ds = TaskDataset(train_data, root=TRAIN_PATH, task=2)
        train_dl = DataLoader(train_ds, batch_size=50, num_workers=4, drop_last=True, shuffle=True)
        eval_ds = TaskDataset(eval_data, root=TRAIN_PATH, task=2)
        eval_dl = DataLoader(eval_ds, batch_size=50, num_workers=4, drop_last=True, shuffle=True)
    else:
        train_ds = TaskDataset(train_data, root=TRAIN_PATH, task=3)
        train_dl = DataLoader(train_ds, batch_size=50, num_workers=4, drop_last=True, shuffle=True)
        eval_ds = TaskDataset(eval_data, root=TRAIN_PATH, task=3) 
        eval_dl = DataLoader(eval_ds, batch_size=50, num_workers=4, drop_last=True, shuffle=True)
    
    for epoch in range(epochs[task]):
        print(f"Epoch [{epoch}]")
        train_loss = 0
        train_acc = 0
        model.train()
        for data in train_dl:
            image, label = data
            image = image.to(device)
            label = label.to(device) #(batch, 62/124/248)
            
            pred = model(image) #(batch, 62/124/248)
            loss = loss_fn(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            cpu_pred = pred.cpu().data.numpy()
            cpu_label = label.cpu().data.numpy()
            
            for i in range(pred.shape[0]):
                l = np.where(cpu_label[i] == 1)[0]
                p=[]
                accuracy = 0
                for j in range(l.shape[0]):
                    p.append(np.argmax(cpu_pred[i, j*62:(j+1)*62])+j*62)
                    
                    if p[j] == l[j]:
                        accuracy += 1
                train_acc += accuracy // l.shape[0]

                # with open('train.txt', 'a') as txt:
                #     txt.write(f'{epoch}, {i}\n{l}\n{p}\n\n')
                
        train_acc_history.append((train_acc/len(train_ds.data)))
        train_loss_history.append(train_loss/len(train_dl))

        eval_loss = 0
        eval_acc = 0
        
        model.eval()
        with torch.no_grad():
            for data in eval_dl:
                image, label = data
                image = image.to(device)
                label = label.to(device)
                
                pred = model(image)
                loss = loss_fn(pred, label)
                
                eval_loss += loss.item()
                
                cpu_pred = pred.cpu().data.numpy()
                cpu_label = label.cpu().data.numpy()
                
                for i in range(pred.shape[0]):
                    l = np.where(cpu_label[i] == 1)[0]
                    p=[]
                    accuracy = 0
                    for j in range(l.shape[0]):
                        p.append(np.argmax(cpu_pred[i, j*62:(j+1)*62])+j*62)
                        
                        if p[j] == l[j]:
                            accuracy += 1
                    eval_acc += accuracy // l.shape[0]

            eval_acc_history.append((eval_acc/len(eval_ds.data)))
            eval_loss_history.append(eval_loss/len(eval_dl))
            
            if eval_acc_history[epoch] > max_valid_acc:
                torch.save(model.state_dict(), f"task{task}_weight_100.pt")
                max_valid_acc = eval_acc_history[epoch]
                global_acc[task] = max_valid_acc
                print("============model is saved============")
                
            
            # print("accuracy (validation):", eval_acc)
        print(f'train loss: {train_loss/len(train_dl)}, train_acc: {train_acc/len(train_ds.data)}')
        print(f'eval loss: {eval_loss/len(eval_dl)}, eval_acc: {eval_acc/len(eval_ds.data)}')

def plot_curve(task):
    plt.subplot(1,2,1)
    e = [i for i in range(epochs[task])]
    plt.plot(e, train_acc_history, label='train')
    plt.plot(e, eval_acc_history, label='eval')
    plt.title(f"task{i}_accuracy")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(e, train_loss_history, label='train')
    plt.plot(e, eval_loss_history, label='eval')
    plt.title(f"task{i}_loss")
    plt.legend()
    
    plt.show() 
        
train_data = []
eval_data = []
epochs = [0, 50, 150, 50]
lr = [0, 1e-3, 1e-3, 1e-3]

if __name__ == '__main__':
    
    load_csv()
    
    for i in range(3, 4):
        print(i)
        if i == 1:
            weights = torch.load("task1_weight.pt") if os.path.exists("task1_weight.pt") else None
            model = models.densenet201(num_classes=62).to(device)
            if weights != None:
                print("train with weight1")
                model.load_state_dict(weights)
        elif i == 2:
            weights = torch.load("task2_weight.pt") if os.path.exists("task2_weight.pt") else None
            model = models.densenet201(num_classes=124).to(device)
            if weights != None:
                print("train with weight2")
                model.load_state_dict(weights)
        else:
            weights = torch.load("task3_weight_3.pt") if os.path.exists("task3_weight_3.pt") else None
            # model = models.densenet201(num_classes=248).to(device)
            model = models.efficientnet_v2_l(num_classes=248).to(device)
            if weights != None:
                print("train with weight3")
                model.load_state_dict(weights)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=lr[i])
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr[i], momentum=0.9)
        loss_fn = nn.MultiLabelSoftMarginLoss()
        
        train_loss_history = []
        train_acc_history = []
        eval_loss_history = []
        eval_acc_history = []
        
        global_acc = [0]
        with open("record.txt", 'r') as r:
            for line in r:
                acc = float(line.split(':')[1])
                global_acc.append(acc)
                
        train(i)
        plot_curve(i)
        
        with open('record.txt', 'w') as r:
            for i in range(1,len(global_acc)):
                r.write(f'task{i}:{global_acc[i]}\n')