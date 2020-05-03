#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import array
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
import pandas as pd


# In[6]:


df = pd.read_csv('dataset1.csv')


# In[7]:


df


# In[8]:


df = df[['location','total_cases']]


# In[5]:


df


# In[9]:


lis1 = dict()
for rows in df.itertuples():
    if rows.location not in lis1:
        lis1[rows.location] = []
    lis1[rows.location].append(rows.total_cases)


# In[10]:


X = []
y = []
for key in lis1.keys():
    lis = lis1[key]
    n = len(lis)
    for i in range(0,n-3):
        temp = lis[i:i+3]
        X.append(temp)
        y.append(lis[i+3])


# In[11]:


X = array(X)
X = X.reshape(len(X),1,3)
print(len(y))
X


# In[9]:


# def split_sequence(sequence, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequence)): 
#         end_ix = i + n_steps
#         if end_ix > len(sequence)-1:
#             break
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)


# In[10]:


# raw_seq = [48,63,71,81,91,102,112,126,146,171,198,256,334,403,497,571,657,730,883,1019,1139,1326,1635,2059,2545,3105,3684,4289,4778,5351,5916,6729,7600,8454,9212]
# # choose a number of time steps
# n_steps = 3
# # split into samples
# X, y = split_sequence(raw_seq, n_steps)
# print('X=')
# print(X)
# print('y')
# print(y)
# n_features = 1
# X = X.reshape((X.shape[0], X.shape[1], n_features))


# In[12]:


X


# In[13]:


model = Sequential()
model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(1, 3)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[14]:


model.fit(X, y, epochs=300, verbose=2)


# In[15]:


# a=7600
# b=8454
# c=9212
# for _ in range(10):
#     x_input = array([a, b, c])
  
#     x_input = x_input.reshape((1, n_steps, n_features))
#     yhat = model.predict(x_input, verbose=0)
#     print(yhat)
#     a=b
#     b=c
#     c=yhat
# D=[]
# n=35
# scale=[]
# for i in range(0,9500,500):
#     scale.append(i)
# for i in range(4,(n+1),1):
#     D.append(i)

#plt.figure(figsize=(18,10))
#plt.xlabel("Cases")
#plt.ylabel("day")
#plt.plot(D,y)
#plt.plot(D,yhatd)
#plt.yticks(scale)
import random
X_test = []
y_test = []
indexes = random.sample(range(0,len(y)-1),100)
for i in indexes:
    X_test.append(X[i])
    y_test.append(y[i])
X_test = array(X_test)
X_test.shape


# In[16]:


yhatd=model.predict(X_test)
print(model.summary())
print("k")


# In[17]:


# print(yhatd)
# print(y_test)
# print('MSE=')
from sklearn import metrics
# print(metrics.mean_squared_error(yhatd,y_test))
print ("MSE   ",metrics.mean_squared_error(yhatd,y_test))
print ("RMSE   ",metrics.mean_squared_error(yhatd,y_test)**0.5)

MAE = 0
for i in range(0,len(y_test)):
    MAE = MAE + (abs(y_test[i]-yhatd[i]))
print('MAE   ',MAE/len(y_test))
# Q=X[0:len(X)]
# P=yhatd[0:len(yhatd)]
# r=[]
# for i in range(len(X)):
# 	r.append(i)
# sum=0
# for i in range(len(r)):
#   #print(abs(Q[i]-P[i]))
#   sum=sum+abs(((yhatd[i]-y[i])/y[i]))
#   #print(sum)

#   #print(abs([i]-yhatd[i]))
# #print('a')
# print(sum,len(r))
# print("MAPI", (sum/len(r))*100)


# In[ ]:





# In[ ]:





# In[ ]:




