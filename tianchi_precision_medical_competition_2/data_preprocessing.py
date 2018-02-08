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

# 数据预处理
# 删除无用列
def process_data(d_train, num):
    # 剔除无关特征
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
    
    d_train.drop(['SNP12'],axis=1,inplace=True)
    d_train.drop(['SNP45'],axis=1,inplace=True)
    d_train.drop(['DM家族史'],axis=1,inplace=True)
    d_train.drop(['SNP6'],axis=1,inplace=True)
    d_train.drop(['SNP46'],axis=1,inplace=True)
    d_train.drop(['SNP18'],axis=1,inplace=True)
    
    d_train.drop(['SNP22'],axis=1,inplace=True)
    d_train.drop(['SNP23'],axis=1,inplace=True)
    d_train.drop(['SNP55'],axis=1,inplace=True)
    d_train.drop(['SNP43'],axis=1,inplace=True)
    d_train.drop(['ACEID'],axis=1,inplace=True)
     
#     d_train.drop(['BMI分类'],axis=1,inplace=True)
    d_train.drop(['SNP27'],axis=1,inplace=True)
    d_train.drop(['SNP28'],axis=1,inplace=True)
    d_train.drop(['SNP17'],axis=1,inplace=True)
#     d_train.drop(['SNP14'],axis=1,inplace=True)
#     d_train.drop(['分娩时'],axis=1,inplace=True)
#     d_train.drop(['SNP12'],axis=1,inplace=True)
    
    # 缺失值补充中位数
    d_train.fillna(d_train.median(axis=0),inplace=True)
#     d_train = set_missing(d_train)  
    d_train.drop(['id'],axis=1,inplace=True)
    d_train.info()

    if num==1:
        train_y = d_train['label'].as_matrix()
        d_train.drop(['label'],axis=1,inplace=True)
        train_x = d_train.as_matrix()
        return train_x, train_y
    return d_train.as_matrix()



from sklearn import grid_search
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

n_estimators = 6000

def train_model(flag):
    train_x, train_y = process_data(data_train, 1)
    if flag :
        from sklearn.model_selection import train_test_split
        train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.1,random_state=1)
    else:
        test_x = process_data(data_test, 2)
    
    
    # 参数优化顺序n_estimators、max_depth和min_samples_split、min_samples_split和min_samples_leaf、max_features、subsample、learning_rate和n_estimators
    gdbt=GradientBoostingClassifier(
                    loss='deviance', ##损失函数回归默认为ls（最小二乘法）、lad（绝对值差）、huber（二者结合）、quantile（分位数差），分类默认deviance  deviance具有概率输出的分类的偏差、exponential
                    learning_rate=0.001, # 默认0.1学习率
                    n_estimators=n_estimators, # 迭代训练次数，回归树个数（弱学习器）
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
    
    gdbt.fit(train_x,train_y)
    
    y_pred = gdbt.predict(test_x)
    
    if flag :
        y_predprob = gdbt.predict_proba(test_x)[:,1]

        print("Accuracy : %.5g" % metrics.accuracy_score(test_y, y_pred))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_predprob))
        # average = 5个选项默认None，返回数组为每个分类的得分，macro,weighted,micro（适用样本不均衡）,samples，前三个适用二分类，最后个适用多分类
        print("f1_score : %.5g" % metrics.f1_score(test_y, y_pred, average='weighted'))
        
        test_score = np.zeros((n_estimators,), dtype=np.float64)
        for i, y_pred in enumerate(gdbt.staged_predict(test_x)):
        #     test_score[i] = gdbt.loss_(test_y, y_pred)
            test_score[i] = metrics.f1_score(test_y, y_pred, average='weighted')
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Deviance')
#         plt.plot(np.arange(n_estimators) + 1, gdbt.train_score_, 'b-',label='Training Set Deviance')
        plt.plot(np.arange(n_estimators) + 1, test_score, 'r-',label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
                
        ###############################################################################
        # Plot feature importance
        feature_importance = gdbt.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, data_train.columns[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()
    else:
        pred_df=pd.DataFrame({"0":y_pred})
        pred_df.info()
        pred_df.to_csv('data/result.txt', index=False, header=None, encoding='gbk')
    
train_model(True)
# train_model(False)

