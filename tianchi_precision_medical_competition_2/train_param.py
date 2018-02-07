# -*- coding: utf-8 -*-
#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib  
from sklearn.ensemble import RandomForestClassifier
matplotlib.use('qt4agg')  
from matplotlib.font_manager import *  
#定义自定义字体，文件名从1.b查看系统中文字体中来  
myfont = FontProperties(fname='/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc')  
#解决负号'-'显示为方块的问题  
matplotlib.rcParams['axes.unicode_minus']=False  
# 导入自动美化matplotlib图形
import seaborn as sns
import pandas as pd

data_train = pd.read_csv('data/f_train_20180204.csv', encoding='gbk')
data_test = pd.read_csv('data/f_test_a_20180204.csv', encoding='gbk')

data_train.info()


def set_missing(data):  
    data_SNP22 = data[['SNP22','id']]
    data_SNP23 = data[['SNP23','id']]
    data_SNP54 = data[['SNP54','id']]
    data_SNP55 = data[['SNP55','id']]
    data_ACEID = data[['ACEID','id']]
    data_DM家族史 = data[['DM家族史','id']]
    
#     data.drop(['SNP22','SNP23','SNP54','SNP55','ACEID'],axis=1,inplace=True)
    data.drop(['DM家族史'],axis=1,inplace=True)
    data.fillna(data.median(axis=0),inplace=True)
     
    train = pd.merge(data_DM家族史, data, on='id')
    known_column=train[train.DM家族史.notnull()]
    known_column = known_column.as_matrix()  
    DM家族史=train[train.DM家族史.isnull()].as_matrix()  
    y=known_column[:,0]  
    x=known_column[:,1:]  
    rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
    rf.fit(x,y)  
    predictDM家族史=rf.predict(DM家族史[:,1:])  
    train.loc[train.DM家族史.isnull(),'DM家族史']=predictDM家族史  
    
    
#     train = pd.merge(data_ACEID, data, on='id')
#     known_column=train[train.ACEID.notnull()]
#     known_column = known_column.as_matrix()  
#     unknown_column=train[train.ACEID.isnull()].as_matrix()  
#     y=known_column[:,0]  
#     x=known_column[:,1:]  
#     rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
#     rf.fit(x,y)  
#     predict=rf.predict(unknown_column[:,1:])  
#     train.loc[train.ACEID.isnull(),'ACEID']=predict  
#      
#      
#     train = pd.merge(data_SNP55, train, on='id')
#     known_column=train[train.SNP55.notnull()]
#     known_column = known_column.as_matrix()  
#     unknown_column=train[train.SNP55.isnull()].as_matrix()  
#     y=known_column[:,0]  
#     x=known_column[:,1:]  
#     rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
#     rf.fit(x,y)  
#     predict=rf.predict(unknown_column[:,1:])  
#     train.loc[train.SNP55.isnull(),'SNP55']=predict  
#      
#      
#     train = pd.merge(data_SNP54, train, on='id')
#     known_column=train[train.SNP54.notnull()]
#     known_column = known_column.as_matrix()  
#     unknown_column=train[train.SNP54.isnull()].as_matrix()  
#     y=known_column[:,0]  
#     x=known_column[:,1:]  
#     rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
#     rf.fit(x,y)  
#     predict=rf.predict(unknown_column[:,1:])  
#     train.loc[train.SNP54.isnull(),'SNP54']=predict  
     
     
#     train = pd.merge(data_SNP23, data, on='id')
#     known_column=train[train.SNP23.notnull()]
#     known_column = known_column.as_matrix()  
#     unknown_column=train[train.SNP23.isnull()].as_matrix()  
#     y=known_column[:,0]  
#     x=known_column[:,1:]  
#     rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
#     rf.fit(x,y)  
#     predict=rf.predict(unknown_column[:,1:])  
#     train.loc[train.SNP23.isnull(),'SNP23']=predict  
     
     
#     train = pd.merge(data_SNP22, data, on='id')
#     known_column=train[train.SNP22.notnull()]
#     known_column = known_column.as_matrix()  
#     unknown_column=train[train.SNP22.isnull()].as_matrix()  
#     y=known_column[:,0]  
#     x=known_column[:,1:]  
#     rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
#     rf.fit(x,y)  
#     predict=rf.predict(unknown_column[:,1:])  
#     train.loc[train.SNP22.isnull(),'SNP22']=predict  
    
    return train

# 数据预处理
# 删除无用列
def process_data(d_train, num):
#     d_train.drop(['id'],axis=1,inplace=True)
    d_train.drop(['产次'],axis=1,inplace=True)
    d_train.drop(['RBP4'],axis=1,inplace=True)
    d_train.drop(['SNP10'],axis=1,inplace=True)
    d_train.drop(['SNP21'],axis=1,inplace=True)
    d_train.drop(['SNP3'],axis=1,inplace=True)
     
    d_train.drop(['SNP53'],axis=1,inplace=True)
    d_train.drop(['SNP5'],axis=1,inplace=True)
    d_train.drop(['SNP30'],axis=1,inplace=True)
    d_train.drop(['SNP8'],axis=1,inplace=True)
    d_train.drop(['SNP9'],axis=1,inplace=True)
    d_train.drop(['SNP4'],axis=1,inplace=True)
      
    d_train.drop(['SNP25'],axis=1,inplace=True)
    d_train.drop(['SNP24'],axis=1,inplace=True)
    d_train.drop(['SNP16'],axis=1,inplace=True)
    d_train.drop(['SNP50'],axis=1,inplace=True)
    d_train.drop(['SNP33'],axis=1,inplace=True)
    d_train.drop(['SNP26'],axis=1,inplace=True)
    d_train.drop(['BMI分类'],axis=1,inplace=True)
    

    d_train.drop(['SNP35'],axis=1,inplace=True)
    d_train.drop(['SNP2'],axis=1,inplace=True)
    d_train.drop(['SNP15'],axis=1,inplace=True)
    d_train.drop(['SNP14'],axis=1,inplace=True)
    d_train.drop(['分娩时'],axis=1,inplace=True)
    
#     d_train.drop(['SNP12'],axis=1,inplace=True)
#     d_train.drop(['SNP45'],axis=1,inplace=True)
#     d_train.drop(['DM家族史'],axis=1,inplace=True)
#     d_train.drop(['SNP6'],axis=1,inplace=True)
#     d_train.drop(['SNP46'],axis=1,inplace=True)
#     d_train.drop(['SNP18'],axis=1,inplace=True)
    
    

# f1_score train : 0.6788840527106001
# Accuracy : 0.7
# AUC Score : 0.777375
# f1_score test : 0.69697

# best_params : {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 7000}
# f1_score train : 0.6572047903348434
# Accuracy : 0.69
# AUC Score : 0.775362
# f1_score test : 0.68467


# 650 3 0.01
# f1_score train : 0.6688011390518747
# Accuracy : 0.69
# AUC Score : 0.772544
# f1_score test : 0.6862


#     d_train = set_missing(d_train)
    # 缺失值补充中位数
#     d_train.loc[d_train.DM家族史.isnull(), 'DM家族史'] = 0
    d_train.fillna(d_train.median(axis=0),inplace=True)
    d_train.drop(['id'],axis=1,inplace=True)
    
    if num==1:
        train_y = d_train['label'].as_matrix()
        d_train.drop(['label'],axis=1,inplace=True)
        train_x = d_train.as_matrix()
        return train_x, train_y
    return d_train.as_matrix()


train_x, train_y = process_data(data_train, 1)
# test_x = process_data(data_test, 2)


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.1,random_state=0)


from sklearn import grid_search
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {
#             "loss":['deviance'],
            "learning_rate":[0.0005],
            "n_estimators":[10000,12000,8000,14000],
#             "max_features":['sqrt'], 
            "max_depth":[3], 
#             "min_samples_split": [200], 
#             "min_samples_leaf":[50], 
#             "subsample":[0.8], 
#             "min_impurity_split":[1e-7],
#             "criterion":['friedman_mse','mse']
              }



# 参数优化顺序n_estimators、max_depth和min_samples_split、min_samples_split和min_samples_leaf、max_features、subsample、learning_rate和n_estimators
gdbt=GradientBoostingClassifier(
                loss='deviance', ##损失函数回归默认为ls（最小二乘法）、lad（绝对值差）、huber（二者结合）、quantile（分位数差），分类默认deviance  deviance具有概率输出的分类的偏差
                learning_rate=0.01, # 默认0.1学习率
                n_estimators=650, # 迭代训练次数，回归树个数（弱学习器）
                max_depth=5, # 默认值3最大树深度
                subsample=0.8, # 子采样率，为1表示全部，推荐0.5-0.8默认1
                criterion='friedman_mse', # 判断节点是否继续分裂采用的计算方法
                min_samples_split=200, # 生成子节点所需要的最小样本数，浮点数代表%
                min_samples_leaf=50, # 叶子节点所需要的最小样本数，浮点数代表%
                min_weight_fraction_leaf=0., # 
                min_impurity_split=1e-7, # 停止分裂叶子节点的阀值
                init=None, # 
                random_state=0, # 随机种子，方便重现
                max_features='sqrt', # 寻找最佳分割点要考虑的特征数量auto和None全选/sqrt开方/log2对数/int自定义个数/float%
                verbose=0,
                max_leaf_nodes=10, # 叶子节点的个数，None不限数量 0.8586
                warm_start=False, # True在前面基础上增量训练（重设参数减少训练次数） False默认值擦除重新训练
                presort='auto')

gdbt = grid_search.GridSearchCV(estimator=gdbt, param_grid=param_grid, n_jobs=1, cv=10, verbose=10, scoring='f1',error_score="raise")
gdbt.fit(train_x,train_y)

print("best_params : "+str(gdbt.best_params_))
print("f1_score train : "+str(gdbt.best_score_))

y_pred = gdbt.predict(test_x)
y_predprob = gdbt.predict_proba(test_x)[:,1]

print("Accuracy : %.5g" % metrics.accuracy_score(test_y, y_pred))
print("AUC Score : %f" % metrics.roc_auc_score(test_y, y_predprob))
# average = 5个选项默认None，返回数组为每个分类的得分，macro,weighted,micro（适用样本不均衡）,samples，前三个适用二分类，最后个适用多分类
print("f1_score test : %.5g" % metrics.f1_score(test_y, y_pred, average='macro'))


