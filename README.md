## 提高二分类任务的准确度 #313

提高准确度的问题，往往有以下方法：
1. 更换更好的模型
2. 调整目前模型的参数
3. 数据清洗，获得更强的特征

经过我的测试，后两种没有明显地提升准确率，因此，我们进行第一种方法，更换更加好的模型。

我测试使用了Lightgbm模型，最后准确率提升至86.6%，提升了大概2个百分点。

整个代码如下：

```
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier

train=np.load('Data/ch14.Income.train.npz')
test=np.load('Data/ch14.Income.test.npz')

train_x = train['data']
train_y = train['label']

test_x = test['data']
test_y = test['label']
test_y = test_y.flatten()

#训练集10%用作验证集
num_val= int(0.1* train_x.shape[0])

val_x = train_x[0:num_val,:]
val_y = train_y[0:num_val,:]
val_y = val_y.flatten()

train_x = train_x[num_val:,:]
train_y = train_y[num_val:,:]
train_y =train_y.flatten()

model = LGBMClassifier(random_state=5, num_leaves=50)

print('fitting...')
model.fit(train_x, train_y)
print('fitting done...')

# 验证集
val_predict = model.predict(val_x)
r = (val_predict == val_y)
val_accuracy = r.sum()/val_x.shape[0]
print(val_accuracy) 

# 测试集
test_predict = model.predict(test_x)
r = (test_predict == test_y)
test_accuracy = r.sum()/test_x.shape[0]
print(test_accuracy)
```

输出结果：

```
fitting...
fitting done...
0.8597480106100795
0.8662018592297477
```

