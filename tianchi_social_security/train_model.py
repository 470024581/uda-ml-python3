#! usr/bin/python
#coding=utf-8
'''
Created on 2017年7月17日

@author: lianglong
'''

import pandas as pd
import numpy as np

train_data = pd.read_csv("data/train.csv", low_memory=False)
test_data = pd.read_csv("data/test.csv", low_memory=False)
id_data = test_data['个人编码']
def process_data(train_data):
#     train_data.info()
    
    # 删除没有数据的列
    train_data.drop(['一次性医用材料拒付金额','农民工医疗救助计算金额','残疾军人医疗补助基金支付金额'],axis=1,inplace=True)
    # 缺失少量数据的，以0补充
    train_data.loc[train_data.公务员医疗补助基金支付金额.isnull(), '公务员医疗补助基金支付金额'] = 0
    train_data.loc[train_data.城乡救助补助金额.isnull(), '城乡救助补助金额'] = 0
    train_data.loc[train_data.本次审批金额.isnull(), '本次审批金额'] = 0
    train_data.loc[train_data.补助审批金额.isnull(), '补助审批金额'] = 0
    train_data.loc[train_data.医疗救助医院申请.isnull(), '医疗救助医院申请'] = 0
    train_data.loc[train_data.民政救助补助金额.isnull(), '民政救助补助金额'] = 0
    train_data.loc[train_data.城乡优抚补助金额.isnull(), '城乡优抚补助金额'] = 0
    train_data.loc[train_data.非典补助补助金额.isnull(), '非典补助补助金额'] = 0
    train_data.loc[train_data.家床起付线剩余.isnull(), '家床起付线剩余'] = 0
    train_data.loc[train_data.一次性医用材料自费金额.isnull(), '一次性医用材料自费金额'] = 0
    train_data.loc[train_data.一次性医用材料申报金额.isnull(), '一次性医用材料申报金额'] = 0
    # 缺失少量数据的，以出现频次最多的补充
#     train_data.loc[train_data.出院诊断病种名称.isnull(), '出院诊断病种名称'] = '挂号'
#     train_data.loc[train_data.操作时间.isnull(), '操作时间'] = '01-7月 -16'
    
    # print(train_data.groupby(['操作时间']).head()['操作时间'])
    # print(train_data['操作时间'].describe())
    
    train_data[["医院编码"]] = train_data[["医院编码"]].astype(np.int64)
    train_data[["个人编码"]] = train_data[["个人编码"]].astype(np.int64)
    
    try:
        train_y = train_data["1"].as_matrix()
#         train_data.drop(["个人编码","标记",'操作时间','申报受理时间','出院诊断病种名称','住院开始时间','住院终止时间','交易时间','顺序号'],axis=1,inplace=True)
        train_data.drop(["个人编码","1"],axis=1,inplace=True)
    except KeyError:
        train_y = None
#         train_data.drop(["个人编码",'操作时间','申报受理时间','出院诊断病种名称','住院开始时间','住院终止时间','交易时间','顺序号'],axis=1,inplace=True)
        train_data.drop(["个人编码"],axis=1,inplace=True)
    train_data.info()
    train_x = train_data.as_matrix()
    return train_x, train_y

# print(test_data['个人编码'])

train_x, train_y = process_data(train_data)
test_x, test_y = process_data(test_data)
from sklearn.model_selection import train_test_split  
x_tr,x_te,y_tr,y_te=train_test_split(train_x,train_y,test_size=0.2,random_state=0)  

# 逻辑回归
# from sklearn.linear_model import LogisticRegression  
# lr=LogisticRegression(C=1.0,tol=1e-6)  
# lr.fit(x_tr,y_tr)  
# print(lr.score(x_te,y_te))

# 高斯贝叶斯
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(x_tr,y_tr)
# print(gnb.score(x_te,y_te))

# 支持向量聚类
# from sklearn.svm import SVC  
# param_grid = {'C':[1,2,3,4], 'kernel':['rbf','sigmoid','poly']}
# svc=SVC(C=2, kernel='rbf', tol=1e-6)
# model = grid_search.GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=1, cv=10, verbose=20)
# svc.fit(x_tr,y_tr)  
# print(svc.score(x_te,y_te))  

# 随机森林
# from sklearn.ensemble import RandomForestClassifier  
# randomf=RandomForestClassifier(n_estimators=500,max_depth=6,random_state=0)  
# randomf.fit(x_tr,y_tr)  
# print(randomf.score(x_te,y_te))  

# 梯度提升分类
from sklearn import grid_search
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
param_grid = {
#             "learning_rate":[0.1],
#               "n_estimators":[200],
#             "max_features":[45,50], 
#             "max_depth":[5,7], 
#               "min_samples_split": [100,200,300], 
#               "min_samples_leaf":[50,75,100], 
#               "subsample":[0.6,0.7,0.8,0.9,1], 
#               "min_impurity_split":[1e-8,1e-7,1e-6]
              }
# 参数优化顺序n_estimators、max_depth和min_samples_split、min_samples_split和min_samples_leaf、max_features、subsample、learning_rate和n_estimators
gdbt=GradientBoostingClassifier(
                loss='deviance', ##损失函数默认deviance  deviance具有概率输出的分类的偏差
                learning_rate=0.005, # 默认0.1学习率
                n_estimators=4000, # 迭代训练次数，回归树个数（弱学习器）
                max_depth=15, # 默认值3最大树深度
                subsample=0.7, # 子采样率，为1表示全部，推荐0.5-0.8默认1
                criterion='friedman_mse', # 判断节点是否继续分裂采用的计算方法
                min_samples_split=300, # 生成子节点所需要的最小样本数，浮点数代表百分比
                min_samples_leaf=100, # 叶子节点所需要的最小样本数，浮点数代表百分比
                min_weight_fraction_leaf=0.,
                min_impurity_split=1e-7, # 停止分裂叶子节点的阀值
                init=None,
                random_state=0, # 随机种子，方便重现，没有明白什么 
                max_features=50, # 寻找最佳分割点要考虑的特征数量auto和None全选/sqrt开方/log2对数/int自定义个数/float百分比
                verbose=0,
                max_leaf_nodes=None, # 叶子节点的个数，None不限数量
                warm_start=False, # True在前面基础上增量训练（重设参数减少训练次数） False默认值擦除重新训练
                presort='auto')  
# 网格参数配合交叉验证n_jobs=模型并行度1，cv=10交叉验证的测试集为10折，verbose=10表示前10个子模型打印日志
gdbt = grid_search.GridSearchCV(estimator=gdbt, param_grid=param_grid, n_jobs=1, cv=5, verbose=10)
gdbt.fit(x_tr,y_tr)
y_pred = gdbt.predict(x_te)
y_predprob = gdbt.predict_proba(x_te)[:,1]
# print("Score : %.5g" % gdbt.score(x_te,y_te))
print("Accuracy : %.5g" % metrics.accuracy_score(y_te, y_pred))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_te, y_predprob))
# average = 5个选项默认None，返回数组为每个分类的得分，macro,weighted,micro（适用样本不均衡）,samples，前三个适用二分类，最后个适用多分类
print("f1_score : %.5g" % metrics.f1_score(y_te, y_pred, average='micro'))
# 
print(gdbt.best_estimator_)
print(gdbt.grid_scores_)
print(gdbt.best_params_)
print(gdbt.best_score_)
# 
predictions=gdbt.predict(test_x).astype(np.int32)  
# print(predictions)
train_data=pd.DataFrame({"0":id_data.as_matrix(),'1':predictions})  
# train_data.info()
train_data.to_csv('data/result2.csv', index=False, header=None)  

