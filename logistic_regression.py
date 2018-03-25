
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import log_loss

train_file_path = r'D:\Data\CVR\round1_ijcai_18_train_20180301.txt'
test_file_path = r'D:\Data\CVR\round1_ijcai_18_test_a_20180301.txt'

def read_data(file_path):
    df = pd.read_table(file_path, sep =  ' ')
    data = df.select_dtypes(include=['int64', 'float64'])
    return data

# Load the dataset
df_train = read_data(train_file_path)
df_train.describe()

df_test = read_data(test_file_path)
df_test.describe()

df_train_count = len(df_train.index)
data_train = df_train[: int(df_train_count * 0.7)]
data_test = df_train[int(df_train_count * 0.7) :]
train_x = data_train.drop(['is_trade'], 1)
train_y = data_train[['is_trade']]
test_x = data_test.drop(['is_trade'], 1)
test_y = data_test[['is_trade']]
train_x.describe()

# Create logistic regression object
regr = linear_model.LogisticRegression()

# Train the model using the training sets
regr.fit(train_x, train_y.values.ravel())

# Make predictions using the testing set
pred_y = regr.predict_proba(test_x)

print('log_loss score: %.2f' % log_loss(test_y, pred_y))

# Plot outputs
plt.scatter(test_x, test_y,  color='black')
plt.plot(test_x, pred_y, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import log_loss

train_file_path = r'D:\Data\CVR\round1_ijcai_18_train_20180301.txt'
test_file_path = r'D:\Data\CVR\round1_ijcai_18_test_a_20180301.txt'

def read_data(file_path):
    df = pd.read_table(file_path, sep =  ' ')
    data = df.select_dtypes(include=['int64', 'float64'])
    return data

# Load the dataset
df_train = read_data(train_file_path)
df_train.describe()

df_test = read_data(test_file_path)
df_test.describe()

df_train_count = len(df_train.index)
data_train = df_train
data_test = df_test
train_x = data_train.drop(['is_trade'], 1)
train_y = data_train[['is_trade']]
test_x = data_test
test_y = None
train_x.describe()

# Create logistic regression object
regr = linear_model.LogisticRegression()

# Train the model using the training sets
regr.fit(train_x, train_y.values.ravel())

# Make predictions using the testing set
pred_y = regr.predict_proba(test_x)

pred_res = pd.DataFrame(test_x['instance_id'])
pred_res['predicted_score'] = pred_y[: , 1]


# In[31]:


dst_path =  r'D:\Data\CVR\round1_ijcai_18_pred_20180326.csv'
pred_res.to_csv(dst_path, sep = ' ', index  = False)


# In[28]:


pred_res.columns

