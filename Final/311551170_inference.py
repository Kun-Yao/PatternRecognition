# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:28:21 2023

@author: kunyao
"""

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

TEST_PATH = "../dataset/test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_onehot(text, words):
    onehot = np.zeros(62*words)
    for i in range(words):
        onehot[alphabets2index[text[i]] + i*62] = 1
    return onehot

def load_csv():      
    data = pd.read_csv(f'{TEST_PATH}/../sample_submission.csv').to_numpy()
    for row in data:
        test_data.append(row)

class TaskDataset(Dataset):
    def __init__(self, data, root, return_filename=False, task=1):
        self.data = [sample for sample in data if sample[0].startswith(f'task{task}')]
        # self.data = [sample for sample in data]
        self.return_filename = return_filename
        self.root = root
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                                         transforms.RandomRotation(30)])
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
        # img = cv2.resize(img, (64, 64))
        
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = img.transpose(2, 0, 1)
        # img = np.mean(img, axis=2)
        # print(f'img.size, {img.size}')
        img_pil = self.trans(img_pil)
        
        if self.return_filename:
            return torch.FloatTensor(img), filename
        else:
            return torch.FloatTensor(img), torch.tensor(to_onehot(label, 2 ** self.task-1))

    def __len__(self):
        return len(self.data) 
    
def test(task):
    print(f'test task{task}')
    model.eval()
    if task == 1:
        test_dl = t1_test_dl
    elif task == 2:
        test_dl = t2_test_dl
    else:
        test_dl = t3_test_dl
    
    
    with torch.no_grad():
        for data in test_dl:
            image, filename = data
            image = image.to(device)

            pred = model(image) #(batch, 62/124/248)
            cpu_pred = pred.cpu().data.numpy()
            for i in range(pred.shape[0]):
                p = []
                pred_char = ''
                for j in range(2 ** (task-1)):
                    p.append(np.argmax(cpu_pred[i, j*62:(j+1)*62])+j*62)
                    pred_char += alphabets[p[j] % 62]
                # print(f'{filename[i]} {pred_char}\n') 
                submission[filename[i]] = pred_char

test_data = []    
if __name__ == '__main__':
    load_csv()
    
    t1_test_ds = TaskDataset(test_data, root=TEST_PATH, return_filename=True, task=1)
    t2_test_ds = TaskDataset(test_data, root=TEST_PATH, return_filename=True, task=2)
    t3_test_ds = TaskDataset(test_data, root=TEST_PATH, return_filename=True, task=3)
    t1_test_dl = DataLoader(t1_test_ds, batch_size=100, num_workers=4, drop_last=False, shuffle=False)
    t2_test_dl = DataLoader(t2_test_ds, batch_size=100, num_workers=4, drop_last=False, shuffle=False)
    t3_test_dl = DataLoader(t3_test_ds, batch_size=100, num_workers=4, drop_last=False, shuffle=False)

    df = pd.read_csv("../dataset/sample_submission.csv").to_numpy()
    submission = dict()
    for i in range(1, 4):
        
        
        if i == 1:
            weights = torch.load("task1_weight.pt") if os.path.exists("task1_weight.pt") else None
            model = models.densenet201(num_classes=62).to(device)
            if weights != None:
                print("test with weights")
                model.load_state_dict(weights)    
        elif i == 2:
            weights = torch.load("task2_weight.pt") if os.path.exists("task2_weight.pt") else None
            model = models.densenet201(num_classes=124).to(device)
            if weights != None:
                print("test with weights")
                model.load_state_dict(weights)
        else:
            weights = torch.load("task3_weight_3.pt") if os.path.exists("task3_weight_3.pt") else None
            model = models.efficientnet_v2_l(num_classes=248).to(device)
            if weights != None:
                print("test with weights")
                model.load_state_dict(weights)
        
        test(task=i)
    submission_df = pd.DataFrame({'filename': submission.keys(), 'label': submission.values()})
    submission_df.to_csv('submission.csv', index=False)