import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

train=np.load('Data/ch14.Income.train.npz')
test=np.load('Data/ch14.Income.test.npz')

train_x = train['data']
train_y = train['label']

test_x = test['data']
test_y = test['label']
test_y = test_y.flatten()

train_x_norm = np.zeros_like(train_x)
test_x_norm = np.zeros_like(test_x)

for i in range(train_x_norm.shape[1]):
	train_x_norm[:,i] = (train_x[:,i]-min(train_x[:,i]))/(max(train_x[:,i])-min(train_x[:,i]))
	
for i in range(test_x_norm.shape[1]):
	test_x_norm[:,i] = (test_x[:,i]-min(test_x[:,i]))/(max(test_x[:,i])-min(test_x[:,i]))


num_val= int(0.1* train_x.shape[0])

val_x = train_x[0:num_val,:]
val_y = train_y[0:num_val,:]
val_y = val_y.flatten()

train_x = train_x[num_val:,:]
train_y = train_y[num_val:,:]
train_y =train_y.flatten()

model = LGBMClassifier(random_state=5, num_leaves=50) #, learning_rate=0.07

print('fitting...')
model.fit(train_x, train_y)
print('fitting done...')

# validation
val_predict = model.predict(val_x)
r = (val_predict == val_y)
val_accuracy = r.sum()/num_val
print(val_accuracy) 

# test
test_predict = model.predict(test_x)
r = (test_predict == test_y)
test_accuracy = r.sum()/test_x.shape[0]
print(test_accuracy)  
