# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:43:35 2023

@author: Lab639
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.DataFrame(pd.read_csv("./PR_HW2_train.csv"))
df_val   = pd.DataFrame(pd.read_csv("./PR_HW2_val.csv"))
df_test  = pd.DataFrame(pd.read_csv("./sample_output.csv"))
df_train.head()
df_test.head()
# Data processing
X_train = df_train[['Feature1', 'Feature2', 'Feature3', 'Feature4']].to_numpy()
y_train = df_train[['Target']].to_numpy()

X_val = df_val[['Feature1', 'Feature2', 'Feature3', 'Feature4']].to_numpy()
y_val = df_val[['Target']].to_numpy()

X_test = df_test[['Feature1', 'Feature2', 'Feature3', 'Feature4']].to_numpy()
y_test = df_test[['Target']].to_numpy()

C1 = X_test[y_test[:,0]==0]
C2 = X_test[y_test[:,0]==1]
C3 = X_test[y_test[:,0]==2]

ax = plt.axes(projection='3d')


ax.scatter(C1[:, 1], C1[:, 2], C1[:, 0], 'r', label='c1')
ax.scatter(C3[:, 1], C3[:, 2], C3[:, 0], 'b', label='c3')
ax.scatter(C2[:, 1], C2[:, 2], C2[:, 0], 'g', label='c2')
# plt.xticks(np.arange(0, 10, 0.2))
# plt.yticks(np.arange(200000, 800000, 10000))
plt.legend()

print(np.mean(C1[:,0]))
print(np.mean(C2[:,0]))
print(np.mean(C3[:,0]))

# bx = plt.axes(projection='3d')
# bx.scatter(C1[:, 0], C1[:, 1], C1[:, 2], 'r')
# bx.scatter(C2[:, 0], C2[:, 1], C2[:, 2], 'g')
# bx.scatter(C3[:, 0], C3[:, 1], C3[:, 2], 'b')


