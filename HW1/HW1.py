# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# TODO
# Implement the linear regression using gradient descent by only numpy

class LinearRegression:
    
    def __init__(self):
        
        print("Don't cheat.")
        self.epoch = []
        self.train_loss = []
        self.weights = None

    def fit(self, X, y, lr=0.001, epochs=100, batch_size=1):
        self.weights = np.zeros(((X.shape[1]), 1))
        # self.weights = np.array([(380.0),(1382.0)]).reshape((2,1))
        # self.weights = np.array([(259.0), (-383.0), (333.0), (442.0), (24032.0), (-416.0), (-11857.0)]).reshape((7,1))
        
        for epoch in range(epochs):

            for batch in range(len(X)//batch_size):
                start = batch*batch_size
                end = start + batch_size
                
                # print("X.shape: {}" .format(X.shape))
                # print("y.shape: {}" .format(y.shape))
                
                y_hat = self.predict(X[start:end]).reshape((batch_size,y.shape[1]))
                # print("y_hat.shape: {}" .format(y_hat.shape))
                
                
                dw = (-2/batch_size) * (((X[start:end]).transpose()).dot(y[start:end]-y_hat))
                
                # print("weight.shape: {}" .format(self.weights.shape))
                # print("dw.shape: {}" .format(dw.shape))
                # print((lr * dw).shape)
                self.weights -= (lr * dw)
                
                # print("weights: {}" .format(self.weights))
                # self.weights[:-1] -= lr * -2 * ((X.transpose()).dot((y-y_hat)).sum()/len(X))
                # self.weights[-1] -= lr * -2 * ((y-y_hat).sum()/len(X))
                #print(self.weights[:-1])
                
                # pass
            
            self.epoch.append(epoch)
            self.train_loss.append(self.get_loss(X, y))
            # print(self.get_loss(X, y))
            

    def get_loss(self, X, y):
        y_hat = self.predict(X)
        return (np.square(y-y_hat)).sum() / len(y)
        #pass

    def predict(self, X):
        #y_hat = self.weights[:-1]*X + self.weights[-1]
        # ones = np.ones((X.shape[0], 1))
        # print("ones.shape: {}" .format(ones.shape))
        # X = np.concatenate((X,ones), axis=1)
        # print("pX.shape: {}" .format(X.shape))
        y_hat = X.dot(self.weights)
        return y_hat
        #pass
                
    def evaluate(self, X, y):
        return self.get_loss(X, y)
        #pass
        
    def plot_curve(self):
        # self.epoch and self.train_loss may be helpful here. 
        plt.plot(self.epoch, self.train_loss, label='Train MSE loss')
        plt.title("Training loss")
        plt.legend(loc='upper right')
        plt.show()
        #pass
    
    
# Load data & data pre-processing

df_train = pd.DataFrame(pd.read_csv("./regression_train.csv"))
df_val   = pd.DataFrame(pd.read_csv("./regression_val.csv"))
df_test  = pd.DataFrame(pd.read_csv("./regression_test.csv"))

# df_train.head()

# df_test.head()

# df_train.info()

# TODO
# You may do the labelEncoder here

#### For multiple features, please use the following settings.####
# sex.female -> 0
# sex.male -> 1

# smoker.no -> 0
# smoker.yes -> 1

# region.northeast -> 0
# region.northwest -> 1
# region.southeast -> 2
# region.southwest -> 3
##################################################################
#%%
df_train['sex'] = df_train['sex'].replace('female', 0)
df_train['sex'] = df_train['sex'].replace('male', 1)
df_train['smoker'] = df_train['smoker'].replace('no', 0)
df_train['smoker'] = df_train['smoker'].replace('yes', 1)
df_train['region'] = df_train['region'].replace('northeast', 0)
df_train['region'] = df_train['region'].replace('northwest', 1)
df_train['region'] = df_train['region'].replace('southeast', 2)
df_train['region'] = df_train['region'].replace('southwest', 3)


df_val['sex'] = df_val['sex'].replace('female', 0)
df_val['sex'] = df_val['sex'].replace('male', 1)
df_val['smoker'] = df_val['smoker'].replace('no', 0)
df_val['smoker'] = df_val['smoker'].replace('yes', 1)
df_val['region'] = df_val['region'].replace('northeast', 0)
df_val['region'] = df_val['region'].replace('northwest', 1)
df_val['region'] = df_val['region'].replace('southeast', 2)
df_val['region'] = df_val['region'].replace('southwest', 3)


df_test['sex'] = df_test['sex'].replace('female', 0)
df_test['sex'] = df_test['sex'].replace('male', 1)
df_test['smoker'] = df_test['smoker'].replace('no', 0)
df_test['smoker'] = df_test['smoker'].replace('yes', 1)
df_test['region'] = df_test['region'].replace('northeast', 0)
df_test['region'] = df_test['region'].replace('northwest', 1)
df_test['region'] = df_test['region'].replace('southeast', 2)
df_test['region'] = df_test['region'].replace('southwest', 3)

# You may try different label encoding for training your own model

#%%
# Single feature (using bmi)

# Do not modify here

x_train = df_train.drop(['charges'], axis=1)
y_train = df_train['charges']
x_train = x_train[['bmi']]

x_val = df_val.drop(['charges'], axis=1)
y_val = df_val['charges']
x_val = x_val[['bmi']]

x_test = df_test.drop(['charges'], axis=1)
x_test = x_test[['bmi']]

# TODO
# You may convert data to NumPy here 
x_train = x_train.to_numpy().reshape(x_train.shape)
y_train = y_train.to_numpy().reshape((y_train.shape[0],1))
ones = np.ones((x_train.shape[0], 1))
x_train = np.concatenate((x_train, ones), axis=1)

x_val = x_val.to_numpy().reshape(x_val.shape)
y_val = y_val.to_numpy().reshape((y_val.shape[0],1))
ones = np.ones((x_val.shape[0], 1))
x_val = np.concatenate((x_val, ones), axis=1)

x_test = x_test.to_numpy().reshape(x_test.shape)
ones = np.ones((x_val.shape[0], 1))
x_test = np.concatenate((x_test, ones), axis=1)

batch_size = x_train.shape[0]

# TODO
# Tune the parameters
# Refer to slide page 9
lr = 0.0007
epochs = 200000

linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size)

print("Intercepts: ", linear_reg.weights[-1])
print("Weights: ", linear_reg.weights[:-1])

print('training loss: ', linear_reg.evaluate(x_train, y_train))

print('validation loss: ', linear_reg.evaluate(x_val, y_val))

test_pred = linear_reg.predict(x_test)

linear_reg.plot_curve()

# Use matplotlib to plot the predicted line with the training and validation samples

# TODO
x = np.arange(min(x_train[:,0]),max(x_train[:,0]))
y = linear_reg.weights[:-1]*x + linear_reg.weights[-1]
y = y.transpose()
x = x.reshape(y.shape)
plt.plot(x_train[:,0], y_train, 'b.', label='Training Samples')
plt.plot(x_val[:,0], y_val, 'y.', label='Validation Samples')
plt.plot(x, y, 'r', label='Linear model')
plt.legend(loc='upper left')
plt.show()
#%%

# Multiple features
# Do not modify here

x_train = df_train.drop(['charges'], axis=1)
y_train = df_train['charges']

x_val = df_val.drop(['charges'], axis=1)
y_val = df_val['charges']

x_test = df_test.drop(['charges'], axis=1)

# TODO
# You may convert data to NumPy here 
x_train = x_train.to_numpy().reshape(x_train.shape)
y_train = y_train.to_numpy().reshape((y_train.shape[0],1))
ones = np.ones((x_train.shape[0], 1))
x_train = np.concatenate((x_train, ones), axis=1)

x_val = x_val.to_numpy().reshape(x_val.shape)
y_val = y_val.to_numpy().reshape((y_val.shape[0],1))
ones = np.ones((x_val.shape[0], 1))
x_val = np.concatenate((x_val, ones), axis=1)

x_test = x_test.to_numpy().reshape(x_test.shape)
ones = np.ones((x_val.shape[0], 1))
x_test = np.concatenate((x_test, ones), axis=1)


batch_size = x_train.shape[0]

# TODO
# Tune the parameters
# Refer to slide page 10
lr = 0.00037
epochs = 585000

linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size)

print("Intercepts: ", linear_reg.weights[-1])
print("Weights: ", linear_reg.weights[:-1])

print('training loss: ', linear_reg.evaluate(x_train, y_train))

print('validation loss: ', linear_reg.evaluate(x_val, y_val))

test_pred = linear_reg.predict(x_test)

linear_reg.plot_curve()

#%%

for i in range(6): 
    plt.subplot(2, 3, i+1)
    x = np.arange(min(x_train[:,i]),max(x_train[:,i]))
    # y = linear_reg.weights[:-1]*x + linear_reg.weights[-1]
    # y = y.transpose()
    x = x.transpose()
    plt.plot(x_train[:,i], y_train, 'b.', label='Training Samples')
    plt.plot(x_val[:,i], y_val, 'y.', label='Validation Samples')
plt.show()

age = np.concatenate((x_train[:, 0], x_val[:, 0]))
bmi = np.concatenate((x_train[:, 2], x_val[:, 2]))
smoke = np.concatenate((x_train[:, 4], x_val[:, 4]))
charge = np.concatenate((y_train, y_val))

plt.subplot(1, 2, 1)
plt.ylabel("charges")
plt.xlabel("age")
for i in range(len(age)):
    if smoke[i] == 1:
        plt.plot(age[i], charge[i], 'r.')
    else:
        plt.plot(age[i], charge[i], 'g.')
        
plt.subplot(1, 2, 2)
plt.xlabel("BMI")
for i in range(len(bmi)):
    if smoke[i] == 1:
        plt.plot(bmi[i], charge[i], 'r.')
    else:
        plt.plot(bmi[i], charge[i], 'g.')
        
# Train your own model and predict for testing data.
def smoker_bmi(s,b):
    sb = np.zeros(s.shape)
    for i in range(len(s)):
        if s[i]==0:
            sb[i]=0
        elif b[i]<30:
            sb[i]=1
        else:
            sb[i]=2
    return sb

x_train = df_train.drop(['charges'], axis=1)
# x_train = x_train.drop(['region'], axis=1)
# x_train = x_train.drop(['sex'], axis=1)
x_train['smoker_bmi'] = smoker_bmi(x_train['smoker'], x_train['bmi'])
x_train = x_train.drop(['smoker'], axis=1)
x_train = x_train.drop(['bmi'], axis=1)
y_train = df_train['charges']

x_val = df_val.drop(['charges'], axis=1)
# x_val = x_val.drop(['region'], axis=1)
# x_val = x_val.drop(['sex'], axis=1)
x_val['smoker_bmi'] = smoker_bmi(x_val['smoker'], x_val['bmi'])
x_val = x_val.drop(['smoker'], axis=1)
x_val = x_val.drop(['bmi'], axis=1)
y_val = df_val['charges']

x_test = df_test.drop(['charges'], axis=1)
# x_test = x_test.drop(['region'], axis=1)
x_test['smoker_bmi'] = smoker_bmi(x_test['smoker'], x_test['bmi'])
x_test = x_test.drop(['smoker'], axis=1)
x_test = x_test.drop(['bmi'], axis=1)
# x_test = x_test.drop(['sex'], axis=1)

x_train = x_train.to_numpy().reshape(x_train.shape)
y_train = y_train.to_numpy().reshape((y_train.shape[0],1))
ones = np.ones((x_train.shape[0], 1))
x_train = np.concatenate((x_train, ones), axis=1)

x_val = x_val.to_numpy().reshape(x_val.shape)
y_val = y_val.to_numpy().reshape((y_val.shape[0],1))
ones = np.ones((x_val.shape[0], 1))
x_val = np.concatenate((x_val, ones), axis=1)

x_test = x_test.to_numpy().reshape(x_test.shape)
ones = np.ones((x_val.shape[0], 1))
x_test = np.concatenate((x_test, ones), axis=1)



#%%

batch_size = 50
lr = 0.00007
epochs = 100000

linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size)

print('training loss: ', linear_reg.evaluate(x_train, y_train))

print('validation loss: ', linear_reg.evaluate(x_val, y_val))

linear_reg.plot_curve()

test_pred = linear_reg.predict(x_test)
print("test_pred shape: ", test_pred.shape)
assert test_pred.shape == (200, 1)
#%%


df_test = pd.DataFrame(pd.read_csv("./regression_test.csv"))
df_test["charges"] = test_pred
# df_test.to_csv("sample_output.csv")