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


train = pd.read_csv('data/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('data/d_test_A_20180102.csv', encoding='utf8')


# 查看缺失值比例
data = pd.concat([train,test],axis=0)
print(data.isnull().sum()/len(data))

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# train['乙肝e抗体'] = train['乙肝e抗体'].dropna()
# fig ,ax = plt.subplots()
# ax.scatter(x = train['乙肝e抗体'],y=train['血糖'])
# plt.ylabel('血糖')
# plt.xlabel('乙肝e抗体')

train.drop(train[(train.血糖>=20)].index,inplace=True)


# 血糖 is the variable we need to predict. So let's do some analysis on this variable first.
sns.distplot(train['血糖'],fit=norm)
(mu,sigma) = norm.fit(train['血糖'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('血糖分布')
 
fig = plt.figure()
res = stats.probplot(train['血糖'], plot=plt)
plt.show()
