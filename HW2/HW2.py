# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:00:32 2023

@author: Lin
"""


# Only these three packages are allowed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
class FLD:
    def __init__(self):
        """
            You can add/change any variables/methods to meet your need.
        """
        self.mean_vectors = None
        self.sw = None
        self.sb = None
        self.w = None
        self.slope = None
        
    def fit(self, X, y):
        # Separate the data into three classes
        C1 = X[y[:,0]==0]
        C2 = X[y[:,0]==1]
        C3 = X[y[:,0]==2]
        # print(C3.shape)
        
        # Calculate the mean vectors of the three classes
        m1 = np.mean(C1, axis=0)
        m2 = np.mean(C2, axis=0)
        m3 = np.mean(C3, axis=0)
        self.mean_vectors = np.array([m1, m2, m3])
        
        # Compute the within-class scatter matrix self.sw
        self.sw = np.zeros((X.shape[1], X.shape[1]))
        self.sw += (C1-m1).T @ (C1-m1)
        self.sw += (C2-m2).T @ (C2-m2)
        self.sw += (C3-m3).T @ (C3-m3)
        print("sw", self.sw.shape)
        # Compute the between-class scatter matrix self.sb
        self.sb = np.zeros((X.shape[1], X.shape[1]))
        self.sb += C1.shape[0] * (m1 - np.mean(X, axis=0)) @ (m1 - np.mean(X, axis=0)).T
        self.sb += C2.shape[0] * (m2 - np.mean(X, axis=0)) @ (m2 - np.mean(X, axis=0)).T
        self.sb += C3.shape[0] * (m3 - np.mean(X, axis=0)) @ (m3 - np.mean(X, axis=0)).T
        print("sb", self.sb.shape)
        # Compute the eigenvalues and eigenvectors of self.sw^-1 * self.sb
        sw_inv = np.linalg.inv(self.sw)
        M = sw_inv @ self.sb
        eigenvalues, eigenvectors = np.linalg.eig(M)
        
        
        indices = np.argmin(eigenvalues)
        eigenvalues = eigenvalues[indices]
        self.w = eigenvectors[indices]
        print("eigens", eigenvalues.shape, eigenvectors.shape)
        
        print('w', self.w.shape)
        
        # pass
        
    def predict_using_class_mean(self, X, y, X_test):
        proj_mean = self.mean_vectors @ self.w.reshape((-1, 1))
        y_pred = X_test @ self.w.reshape((-1, 1))
        d = np.ones((self.mean_vectors.shape[0], 1))
        for i in range(len(y_pred)):
            d[0] = (y_pred[i,0] - proj_mean[0]) ** 2
            d[1] = (y_pred[i,0] - proj_mean[1]) ** 2
            d[2] = (y_pred[i,0] - proj_mean[2]) ** 2
            y_pred[i,0] = np.argmin(d, axis = 0)
        # print(y_pred)
        return y_pred
        pass

    def predict_using_knn(self, X, y, X_test, k=1):
        y.astype('int32')
        y_pred = X_test @ self.w.reshape((-1, 1))
        proj_X = X @ self.w.reshape((-1,1))
        dist = np.zeros((X.shape[0],1))
        for i in range(X_test.shape[0]):
            dist = (y_pred[i,0] - proj_X[:,0]) ** 2
            dist = np.argsort(dist, axis = 0)
            cnt = np.array([0,0,0])
            for j in range(k):
                cnt[y[int(dist[k]), 0]] += 1
            y_pred[i,0] = np.argmax(cnt)
        # print(y_pred.reshape((1,-1)))
        return y_pred
        pass

    def show_confusion_matrix(self, y, y_pred):
        m = np.zeros((3,3))
        for i in range(y.shape[0]):
            m[y[i,0], int(y_pred[i,0])] += 1
        print(m)
        pass

    def plot_projection(self, X, y):
        C1 = X[y[:,0]==0]
        C2 = X[y[:,0]==1]
        C3 = X[y[:,0]==2]
        proj = (X @ self.w).reshape((-1,1)) * self.w / (self.w @ self.w)
        proj_c1 = proj[y[:,0] == 0]
        proj_c2 = proj[y[:,0] == 1]
        proj_c3 = proj[y[:,0] == 2]
        print(proj_c2.shape)
        
        #space
        up = np.max(proj[:, 0]) + 1
        low = np.min(proj[:, 0]) - 1
        x = [low, up]
        self.slope = self.w[1] / self.w[0]
        y = [self.slope*x[0], self.slope*x[1]]
        
        #line
        plt.plot(x, y, lw=1, c='k')
        
        #data
        plt.plot(C1[:,0], C1[:,1], 'r.', label="class 0")
        plt.plot(C2[:,0], C2[:,1], 'g.', label="class 1")
        plt.plot(C3[:,0], C3[:,1], 'b.', label="class 2")
        
        #projected data
        plt.plot(proj_c1[:,0], proj_c1[:,1], 'r.')
        plt.plot(proj_c2[:,0], proj_c2[:,1], 'g.')
        plt.plot(proj_c3[:,0], proj_c3[:,1], 'b.')
        
        for i in range(X.shape[0]):
            plt.plot([X[i,0], proj[i,0]], [X[i,1], proj[i,1]], lw=0.1, c='pink')
        
        plt.title("Projection line: w={}, b=0" .format(self.w))
        plt.legend(loc=0)
        pass

    def accuracy_score(self, y, y_pred):
        cnt = 0
        for i in range(y.shape[0]):
            if y[i,0] == int(y_pred[i,0]):
                cnt += 1
        return cnt / y.shape[0]
        pass
#%%
class MultiClassLogisticRegression:
    
    def __init__(self):
        """
            You can add/change any variables/methods to meet your need.
        """
        self.epoch = []
        self.train_loss = []
        self.weights = None
        self.EPS = 1e-7
        self.val_acc = 0
        self.tmp_w = None

    def fit(self, X, y, batch_size=16, lr=0.001, epoch=100, x_val=None, y_val=None):
        self.tmp_w = np.random.randn(X.shape[1], 3)
        self.bias = np.zeros(3)
           
        y = y.astype(int)
        one_hot = np.zeros((y.shape[0], 3))
        one_hot[range(y.shape[0]), y[:,0]] = 1
        for i in range(epoch):
            for j in range(len(X)//batch_size): 
                start = j * batch_size
                end = start + batch_size
                
                linear = X[start:end] @ self.tmp_w
                pred = self.softmax(linear)
                
                error = (pred - one_hot[start:end]) / y[start:end].shape[0]
                grad_weights = X[start:end].transpose() @ error
                
                self.tmp_w -= grad_weights * lr
                
            if not((x_val==None).all() or (y_val==None).all()):
                acc = self.evaluate(X, y)
                if self.val_acc < acc:
                    self.val_acc = acc
                    self.weights = self.tmp_w
            else:
                self.weights = self.tmp_w
            probs, y_hat = self.predict(X)
            self.train_loss.append(self.cross_entropy(y, probs))
            self.epoch.append(i)
            if self.train_loss[-1] < 1:
                lr = min(lr, 0.9)
            elif self.train_loss[-1] < 0.3:
                break
            
    def predict(self, X):
        probs = self.softmax(X @ self.tmp_w)
        y_hat = np.argmax(probs, axis=1).reshape((-1, 1))
        return probs, y_hat
        #pass

    def evaluate(self, X, y):
        probs, y_hat = self.predict(X)
        return self.accuracy_score(y, y_hat)
        pass

    def softmax(self, z):
        Max = np.max(z,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(z - Max) #subtracts each row with its max value
        Sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / Sum 
        return f_x
        #pass
    
    def cross_entropy(self, y, probs):
        loss = (-1/y.shape[0]) * np.sum(np.log(probs[range(y.shape[0]), y[:,0]] + self.EPS))
        return loss
        #pass
    
    def accuracy_score(self, y, y_pred):
        acc = 0.0
        for i in range(y.shape[0]):
            if int(y[i,0]) == int(y_pred[i]):
                acc += 1
        return acc / y.shape[0]
        pass

    def show_confusion_matrix(self, X, y):
        probs, y_hat = self.predict(X)
        y = y.astype(int)
        m = np.zeros((3,3))
        for i in range(len(y)):
            m[y[i,0], y_hat[i]] += 1
        
        print(m)
        # pass

    def plot_curve(self):
        plt.plot(self.epoch, self.train_loss)
        plt.show()
#%%
        
# Prepare data for Q1 ~ Q12
df_train = pd.DataFrame(pd.read_csv("./PR_HW2_blob_train.csv"))
df_test  = pd.DataFrame(pd.read_csv("./PR_HW2_blob_test.csv"))

X_train = df_train[['Feature1', 'Feature2']].to_numpy()
y_train = df_train[['Target']].to_numpy()

X_test = df_test[['Feature1', 'Feature2']].to_numpy()
y_test = df_test[['Target']].to_numpy()

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test",  X_test.shape)
print("y_test",  y_test.shape)
#%%

# MultiClass Logistic Regression
# For Q1
lr = 0.003
batch_size = 50
epoch = 10000

logistic_reg = MultiClassLogisticRegression()
logistic_reg.fit(X_train, y_train, lr=lr, batch_size=batch_size, epoch=epoch)

# For Q2
print('Training acc: ', logistic_reg.evaluate(X_train, y_train))
# For Q3
print('Valuation acc: ', logistic_reg.evaluate(X_test, y_test))
# For Q4
logistic_reg.plot_curve()
# # For Q5
logistic_reg.show_confusion_matrix(X_test, y_test)

#%%
'''
# Fisher's Linear Discriminant Analysis
fld = FLD()

fld.fit(X_train, y_train)
# For Q6
print("Class mean vector: ", fld.mean_vectors)
# For Q7
print("Within-class scatter matrix SW: ", fld.sw)
# For Q8
print("Between-class scatter matrix SB: ", fld.sb)
# For Q9
print("W: ", fld.w)

# For Q10
y_pred = fld.predict_using_class_mean(X_train, y_train, X_test)
print("FLD using class mean, accuracy: ", fld.accuracy_score(y_test, y_pred))

fld.show_confusion_matrix(y_test, y_pred)
# For Q11
y_pred_k1 = fld.predict_using_knn(X_train, y_train, X_test, k=1)
print("FLD using knn (k=1), accuracy: ", fld.accuracy_score(y_test, y_pred_k1))

y_pred_k2 = fld.predict_using_knn(X_train, y_train, X_test, k=2)
print("FLD using knn (k=2), accuracy: ", fld.accuracy_score(y_test, y_pred_k2))

y_pred_k3 = fld.predict_using_knn(X_train, y_train, X_test, k=3)
print("FLD using knn (k=3), accuracy: ", fld.accuracy_score(y_test, y_pred_k3))

y_pred_k4 = fld.predict_using_knn(X_train, y_train, X_test, k=4)
print("FLD using knn (k=4), accuracy: ", fld.accuracy_score(y_test, y_pred_k4))

y_pred_k5 = fld.predict_using_knn(X_train, y_train, X_test, k=5)
print("FLD using knn (k=5), accuracy: ", fld.accuracy_score(y_test, y_pred_k5))
# For Q12, using only training data
fld.plot_projection(X_train, y_train)
'''
#%%
# Train your own model on provided dataset.
#You can only using 1) **Fisher's Linear Discriminant** or 2) **Logistic Regression** that you have implemented above.
df_train = pd.DataFrame(pd.read_csv("./PR_HW2_train.csv"))
df_val   = pd.DataFrame(pd.read_csv("./PR_HW2_val.csv"))
df_test  = pd.DataFrame(pd.read_csv("./PR_HW2_test.csv"))
df_train.head()
df_test.head()
# Data processing
X_train = df_train[['Feature1', 'Feature2', 'Feature3', 'Feature4']].to_numpy()
y_train = df_train[['Target']].to_numpy()

X_val = df_val[['Feature1', 'Feature2', 'Feature3', 'Feature4']].to_numpy()
y_val = df_val[['Target']].to_numpy()

X_test = df_test[['Feature1', 'Feature2', 'Feature3', 'Feature4']].to_numpy()

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_val",  X_val.shape)
print("y_val",  y_val.shape)
print("X_test", X_test.shape)
'''
for i in range(4):
    plt.plot(X_train[:, i], y_train, '.')
    plt.show()
k=1
for i in range(4):
    for j in range(i+1, 4): 
        plt.subplot(2,3,k)
        ct = np.array([X_train[:,i], X_train[:,j]]).T
        class1 = ct[y_train[:,0] == 0]
        class2 = ct[y_train[:,0] == 1]
        class3 = ct[y_train[:,0] == 2]
        
        plt.plot(class1[:,0], class1[:,1], 'r.')
        plt.plot(class2[:,0], class2[:,1], 'g.')
        plt.plot(class3[:,0], class3[:,1], 'bo', markerfacecolor='none')
        plt.title(f'train ({i}, {j})') 
        plt.axis('off')
        k+=1
plt.show()

for i in range(4):
    plt.plot(X_val[:, i], y_val, '.')
    plt.show()
k=1
for i in range(4):
    for j in range(i+1, 4): 
        plt.subplot(2,3,k)
        cv = np.array([X_val[:,i], X_val[:,j]]).T
        class1 = cv[y_val[:,0] == 0]
        class2 = cv[y_val[:,0] == 1]
        class3 = cv[y_val[:,0] == 2]
        
        plt.plot(class1[:,0], class1[:,1], 'r.')
        plt.plot(class2[:,0], class2[:,1], 'g.')
        #plt.plot(class3[:,0], class3[:,1], 'b.')
        plt.title(f'val ({i}, {j})') 
        plt.axis('off')
        k+=1
plt.show()
'''    
X_train = np.array([X_train[:,1], X_train[:,2]])
X_val = np.array([X_val[:,1], X_val[:,2]])
X_test =np.array([X_test[:,1], X_test[:,2]])
X_train = X_train.T
X_val = X_val.T
X_test = X_test.T
# Refer to section "Prepare data for Q1 ~ Q12"
# Train your model here
# lr = 20
# batch_size = 10
# epoch = 100000
# print(lr, batch_size, epoch)
# your_model = MultiClassLogisticRegression()
# your_model.fit(X_train, y_train, lr=lr, batch_size=batch_size, epoch=epoch, x_val=X_val, y_val=y_val)
# print("last loss", your_model.train_loss[-1])
# # For Q2
# print('Training acc: ', your_model.evaluate(X_train, y_train))
# # For Q3
# print('Valuation acc: ', your_model.evaluate(X_val, y_val))
# # For Q4
# your_model.plot_curve()
# # # For Q5
# your_model.show_confusion_matrix(X_val, y_val)
# test_probs, test_pred = your_model.predict(X_test)
# print("test_pred shape: ", test_pred.shape)

your_model = FLD()
your_model.fit(X_train, y_train)
for i in range(1,6):
    pred_val = your_model.predict_using_knn(X_train, y_train, X_val, k=2)
    acc_val = your_model.accuracy_score(y_val, pred_val)
    print(f'k={i}, acc={acc_val}')

# Output the csv file
# For Q13
df_test = pd.DataFrame(pd.read_csv("./PR_HW2_test.csv"))
df_test["Target"] = test_pred
df_test.to_csv("sample_output.csv")

