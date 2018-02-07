import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew

# 85��1000�����ݣ����Լ�200������
train = pd.read_csv('data/f_train_20180204.csv', encoding='gbk')

train.info()
# print(train.describe())
test = pd.read_csv('data/f_test_a_20180204.csv', encoding='gbk')
# test.info()

# �鿴ȱʧֵ����
data = pd.concat([train,test],axis=0)
print(data.isnull().sum()/len(data))

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # ָ��Ĭ������
mpl.rcParams['axes.unicode_minus'] = False # �������ͼ���Ǹ���'-'��ʾΪ���������

train['hsCRP'] = train['hsCRP'].dropna()
fig ,ax = plt.subplots()
ax.scatter(x = train['hsCRP'],y=train['label'])
plt.ylabel('label')
plt.xlabel('hsCRP')

# label is the variable we need to predict. So let's do some analysis on this variable first.
sns.distplot(train['label'],fit=norm)
(mu,sigma) = norm.fit(train['label'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
 
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('label�ֲ�')
  
fig = plt.figure()
res = stats.probplot(train['label'], plot=plt)
plt.show()

