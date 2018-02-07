# -*- coding: utf-8 -*-
#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib  
matplotlib.use('qt4agg')  
from matplotlib.font_manager import *  
#定义自定义字体，文件名从1.b查看系统中文字体中来  
myfont = FontProperties(fname='/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc')  
#解决负号'-'显示为方块的问题  
matplotlib.rcParams['axes.unicode_minus']=False  
# 导入自动美化matplotlib图形
import seaborn as sns
import pandas as pd

# 合并了测试集a的数据
data_train = pd.read_csv("data/d_train.csv", low_memory=False, encoding='gb2312')
data_test = pd.read_csv("data/d_test_B_20180128.csv", low_memory=False, encoding='gb2312')


# 数据预处理
# 删除无用列
def process_data(d_train, num):
    d_train.drop(['体检日期'],axis=1,inplace=True)
#     d_train['体检日期'] = (pd.to_datetime(d_train['体检日期']) - parse('2017-10-09')).dt.days
    # 删除一行id为580的数据，因为性别为？？，血糖大于20的认为是异常值，（异常值有检测算法）
    if num==1:
        # TODO：计算给定特征的Q1（数据的25th分位点）
        Q1 = np.percentile(d_train['血糖'],25)#25%分位数
        # TODO：计算给定特征的Q3（数据的75th分位点）
        Q3 = np.percentile(d_train['血糖'],75)
        # TODO：使用四分位范围计算异常阶（1.5倍的四分位距）
        step = (Q3-Q1)*1.5
        print(Q1)
        print(Q3)
        print(step)
        min = Q1 - step
        max = Q3 + step
        print(min)
        print(max)
        d_train.drop(d_train[(d_train.id==580)].index,inplace=True)
        d_train.drop(d_train[(d_train.血糖<=3.5)].index,inplace=True)
        d_train.drop(d_train[(d_train.血糖>=17)].index,inplace=True)
    
    # 枚举类型转换成数值类型
    d_train['性别'] = d_train['性别'].map({'男':1,'女':2})
    # 缺失值补充中位数
    d_train.fillna(d_train.median(axis=0),inplace=True)
    
    d_train.drop(['乙肝表面抗原'],axis=1,inplace=True)
    d_train.drop(['乙肝表面抗体'],axis=1,inplace=True)
    d_train.drop(['乙肝e抗原'],axis=1,inplace=True)
    d_train.drop(['乙肝e抗体'],axis=1,inplace=True)
    d_train.drop(['乙肝核心抗体'],axis=1,inplace=True)
    
    d_train.drop(['id'],axis=1,inplace=True)
    
    d_train
    if num==1:
        train_y = d_train['血糖'].as_matrix()
        d_train.drop(['血糖'],axis=1,inplace=True)
#         print(d_train.columns)
        d_train.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34']
#         print(d_train.columns)
        train_x = d_train.as_matrix()
        return train_x, train_y
    d_train.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34']
    return d_train.as_matrix()


train_x, train_y = process_data(data_train, 1)
test_x = process_data(data_test, 2)


# from sklearn.model_selection import train_test_split
# train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.2,random_state=0)


from sklearn import grid_search
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor

# 参数优化顺序n_estimators、max_depth和min_samples_split、min_samples_split和min_samples_leaf、max_features、subsample、learning_rate和n_estimators
gdbt=GradientBoostingRegressor(
                loss='ls', ##损失函数回归默认为ls（最小二乘法）、lad（绝对值差）、huber（二者结合）、quantile（分位数差），分类默认deviance  deviance具有概率输出的分类的偏差
                learning_rate=0.001, # 默认0.1学习率
                n_estimators=6500, # 迭代训练次数，回归树个数（弱学习器）
                max_depth=7, # 默认值3最大树深度
                subsample=0.7, # 子采样率，为1表示全部，推荐0.5-0.8默认1
                criterion='friedman_mse', # 判断节点是否继续分裂采用的计算方法
                min_samples_split=200, # 生成子节点所需要的最小样本数，浮点数代表%
                min_samples_leaf=50, # 叶子节点所需要的最小样本数，浮点数代表%
                min_weight_fraction_leaf=0.,
                min_impurity_split=1e-7, # 停止分裂叶子节点的阀值
                init=None,
                random_state=0, # 随机种子，方便重现
                max_features=0.7, # 寻找最佳分割点要考虑的特征数量auto和None全选/sqrt开方/log2对数/int自定义个数/float%
                verbose=0,
                max_leaf_nodes=10, # 叶子节点的个数，None不限数量 0.8586
                warm_start=False, # True在前面基础上增量训练（重设参数减少训练次数） False默认值擦除重新训练
                presort='auto')

gdbt.fit(train_x,train_y)
y_pred = gdbt.predict(test_x)



# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(test_y, y_pred) * 0.5
# print("MSE: %.4f" % mse)#10000: 1.8054  MSE: 1.8316 MSE: 1.8371
# # compute test set deviance
# test_score = np.zeros((6500,), dtype=np.float64)
# for i, y_pred in enumerate(gdbt.staged_predict(test_x)):
# #     test_score[i] = gdbt.loss_(test_y, y_pred)
#     test_score[i] = mean_squared_error(test_y, y_pred)
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Deviance')
# plt.plot(np.arange(6500) + 1, gdbt.train_score_, 'b-',label='Training Set Deviance')
# plt.plot(np.arange(6500) + 1, test_score, 'r-',label='Test Set Deviance')
# plt.legend(loc='upper right')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Deviance')
#       
# ###############################################################################
# # Plot feature importance
# feature_importance = gdbt.feature_importances_
# # make importances relative to max importance
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, data_train.columns[sorted_idx])
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.show()



pred_df=pd.DataFrame({"0":y_pred})
pred_df.to_csv('data/result.csv', index=False, header=None)

