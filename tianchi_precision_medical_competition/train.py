# -*- coding: utf-8 -*-
#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib  
matplotlib.use('qt4agg')  
from matplotlib.font_manager import *  
#�����Զ������壬�ļ�����1.b�鿴ϵͳ������������  
myfont = FontProperties(fname='/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc')  
#�������'-'��ʾΪ���������  
matplotlib.rcParams['axes.unicode_minus']=False  
# �����Զ�����matplotlibͼ��
import seaborn as sns
import pandas as pd

data_train = pd.read_csv("data/d_train.csv", low_memory=False)
data_test = pd.read_csv("data/d_test_A_20180102.csv", low_memory=False)

# d_train.info()
# print(d_train.head(590))
# print(d_train.describe())
# print(d_train.shape)
# print(d_train['�Ա�'])
# print(d_train['Ѫ��'])

# ����Ԥ����
# ɾ��������
def process_data(d_train, num):
    d_train.drop(['�������'],axis=1,inplace=True)
    # ɾ��һ��idΪ580�����ݣ���Ϊ�Ա�Ϊ����
    if num==1:
        d_train.drop(d_train[(d_train.id==580)].index,inplace=True)
#         d_train.drop(d_train[(d_train.Ѫ��>=20)].index,inplace=True)
        # TODO���������������Q1�����ݵ�25th��λ�㣩
        Q1 = np.percentile(d_train['Ѫ��'],25)#25%��λ��
        # TODO���������������Q3�����ݵ�75th��λ�㣩
        Q3 = np.percentile(d_train['Ѫ��'],75)
        # TODO��ʹ���ķ�λ��Χ�����쳣�ף�1.5�����ķ�λ�ࣩ
        step = (Q3-Q1)*1.5
        print(Q1)
        print(Q3)
        print(step)
        min = Q1 - step
        max = Q3 + step
        print(min)
        print(max)
        d_train.drop(d_train[(d_train.id==580)].index,inplace=True)
#         d_train.drop(d_train[(d_train.Ѫ��>=20)].index,inplace=True)
        d_train.drop(d_train[(d_train.Ѫ��<=min)].index,inplace=True)
        d_train.drop(d_train[(d_train.Ѫ��>=max)].index,inplace=True)
        
    # ת������
    d_train['�ȼ�ϸ���ٷֱ�'] = d_train['�ȼ�ϸ��%']
    d_train.drop(['�ȼ�ϸ��%'],axis=1,inplace=True)
    d_train['����ϸ���ٷֱ�'] = d_train['����ϸ��%']
    d_train.drop(['����ϸ��%'],axis=1,inplace=True)
    d_train['����ϸ���ٷֱ�'] = d_train['����ϸ��%']
    d_train.drop(['����ϸ��%'],axis=1,inplace=True)
    d_train['�ܰ�ϸ���ٷֱ�'] = d_train['�ܰ�ϸ��%']
    d_train.drop(['�ܰ�ϸ��%'],axis=1,inplace=True)
    d_train['������ϸ���ٷֱ�'] = d_train['������ϸ��%']
    d_train.drop(['������ϸ��%'],axis=1,inplace=True)
    
    d_train['_���Ŷ����ᰱ��ת��ø'] = d_train['*���Ŷ����ᰱ��ת��ø']
    d_train.drop(['*���Ŷ����ᰱ��ת��ø'],axis=1,inplace=True)
    d_train['_�����ᰱ��ת��ø'] = d_train['*�����ᰱ��ת��ø']
    d_train.drop(['*�����ᰱ��ת��ø'],axis=1,inplace=True)
    d_train['_��������ø'] = d_train['*��������ø']
    d_train.drop(['*��������ø'],axis=1,inplace=True)
    d_train['_r_�Ȱ�����ת��ø'] = d_train['*r-�Ȱ�����ת��ø']
    d_train.drop(['*r-�Ȱ�����ת��ø'],axis=1,inplace=True)
    d_train['_�ܵ���'] = d_train['*�ܵ���']
    d_train.drop(['*�ܵ���'],axis=1,inplace=True)
    d_train['_�򵰰�'] = d_train['*�򵰰�']
    d_train.drop(['*�򵰰�'],axis=1,inplace=True)
    # ö������ת������ֵ����
    sex_mapping = {'��':1,'Ů':2}
    d_train['�Ա�'] = d_train['�Ա�'].map(sex_mapping)
    # ȱʧ������������λ������
    # ȱʧ14��21��ֵ��һ��17��
    d_train.loc[d_train.��ϸ������.isnull(), '��ϸ������'] = d_train['��ϸ������'].median()
    d_train.loc[d_train.��ϸ������.isnull(), '��ϸ������'] = d_train['��ϸ������'].median()
    d_train.loc[d_train.Ѫ�쵰��.isnull(), 'Ѫ�쵰��'] = d_train['Ѫ�쵰��'].median()
    d_train.loc[d_train.��ϸ��ѹ��.isnull(), '��ϸ��ѹ��'] = d_train['��ϸ��ѹ��'].median()
    d_train.loc[d_train.��ϸ��ƽ�����.isnull(), '��ϸ��ƽ�����'] = d_train['��ϸ��ƽ�����'].median()
    d_train.loc[d_train.��ϸ��ƽ��Ѫ�쵰����.isnull(), '��ϸ��ƽ��Ѫ�쵰����'] = d_train['��ϸ��ƽ��Ѫ�쵰����'].median()
    d_train.loc[d_train.��ϸ��ƽ��Ѫ�쵰��Ũ��.isnull(), '��ϸ��ƽ��Ѫ�쵰��Ũ��'] = d_train['��ϸ��ƽ��Ѫ�쵰��Ũ��'].median()
    d_train.loc[d_train.��ϸ������ֲ����.isnull(), '��ϸ������ֲ����'] = d_train['��ϸ������ֲ����'].median()
    d_train.loc[d_train.ѪС�����.isnull(), 'ѪС�����'] = d_train['ѪС�����'].median() 
    d_train.loc[d_train.ѪС��ƽ�����.isnull(), 'ѪС��ƽ�����'] = d_train['ѪС��ƽ�����'].median()
    d_train.loc[d_train.ѪС������ֲ����.isnull(), 'ѪС������ֲ����'] = d_train['ѪС������ֲ����'].median()
    d_train.loc[d_train.ѪС��Ȼ�.isnull(), 'ѪС��Ȼ�'] = d_train['ѪС��Ȼ�'].median()
    d_train.loc[d_train.������ϸ���ٷֱ�.isnull(), '������ϸ���ٷֱ�'] = d_train['������ϸ���ٷֱ�'].median()
    d_train.loc[d_train.�ܰ�ϸ���ٷֱ�.isnull(), '�ܰ�ϸ���ٷֱ�'] = d_train['�ܰ�ϸ���ٷֱ�'].median() # 26
    d_train.loc[d_train.����ϸ���ٷֱ�.isnull(), '����ϸ���ٷֱ�'] = d_train['����ϸ���ٷֱ�'].median() # 25
    d_train.loc[d_train.����ϸ���ٷֱ�.isnull(), '����ϸ���ٷֱ�'] = d_train['����ϸ���ٷֱ�'].median() # 24
    d_train.loc[d_train.�ȼ�ϸ���ٷֱ�.isnull(), '�ȼ�ϸ���ٷֱ�'] = d_train['�ȼ�ϸ���ٷֱ�'].median() # 23
    
#     d_train.drop(['��ϸ��ѹ��'],axis=1,inplace=True)
#     d_train.drop(['��ϸ��ƽ��Ѫ�쵰����'],axis=1,inplace=True)
#     d_train.drop(['ѪС�����'],axis=1,inplace=True)
#     d_train.drop(['ѪС��ƽ�����'],axis=1,inplace=True)
#     d_train.drop(['ѪС������ֲ����'],axis=1,inplace=True)
#     d_train.drop(['ѪС��Ȼ�'],axis=1,inplace=True)
#     d_train.drop(['�ܰ�ϸ���ٷֱ�'],axis=1,inplace=True)
#     d_train.drop(['����ϸ���ٷֱ�'],axis=1,inplace=True)
#     d_train.drop(['����ϸ���ٷֱ�'],axis=1,inplace=True)
#     d_train.drop(['�ȼ�ϸ���ٷֱ�'],axis=1,inplace=True)
    
    # ȱʧ1219��1221��1378��ֵ���ֱ�4��8��3�У�һ��15��
    # from scipy.stats import mode
    #��mode(d_train['��������'])������
    
    d_train.loc[d_train.��������.isnull(), '��������'] = d_train['��������'].median()
    d_train.loc[d_train.�ܵ��̴�.isnull(), '�ܵ��̴�'] = d_train['�ܵ��̴�'].median()
    d_train.loc[d_train.���ܶ�֬���׵��̴�.isnull(), '���ܶ�֬���׵��̴�'] = d_train['���ܶ�֬���׵��̴�'].median()
    d_train.loc[d_train.���ܶ�֬���׵��̴�.isnull(), '���ܶ�֬���׵��̴�'] = d_train['���ܶ�֬���׵��̴�'].median()
    d_train.loc[d_train._���Ŷ����ᰱ��ת��ø.isnull(), '_���Ŷ����ᰱ��ת��ø'] = d_train['_���Ŷ����ᰱ��ת��ø'].median()
    d_train.loc[d_train._�����ᰱ��ת��ø.isnull(), '_�����ᰱ��ת��ø'] = d_train['_�����ᰱ��ת��ø'].median()
    d_train.loc[d_train._��������ø.isnull(), '_��������ø'] = d_train['_��������ø'].median()
    d_train.loc[d_train._r_�Ȱ�����ת��ø.isnull(), '_r_�Ȱ�����ת��ø'] = d_train['_r_�Ȱ�����ת��ø'].median()
#     d_train.loc[d_train._�ܵ���.isnull(), '_�ܵ���'] = d_train['_�ܵ���'].median() 33
    d_train.loc[d_train.�׵���.isnull(), '�׵���'] = d_train['�׵���'].median()
#     d_train.loc[d_train._�򵰰�.isnull(), '_�򵰰�'] = d_train['_�򵰰�'].median() 34
    d_train.loc[d_train.�������.isnull(), '�������'] = d_train['�������'].median()
    d_train.loc[d_train.����.isnull(), '����'] = d_train['����'].median()
    d_train.loc[d_train.����.isnull(), '����'] = d_train['����'].median()
    d_train.loc[d_train.����.isnull(), '����'] = d_train['����'].median()
    
    d_train.drop(['_�ܵ���'],axis=1,inplace=True)
    d_train.drop(['_�򵰰�'],axis=1,inplace=True)
    
    # ȱʧ�������ݣ������ݲ���Ҫֱ���޳������ݱȽ���Ҫ��ҪС�Ĵ�������ȱʧ���ݲ��ʺ���λ����䣬
#     d_train.loc[d_train.�Ҹα��濹ԭ.isnull(), '�Ҹα��濹ԭ'] = 0
#     d_train.loc[d_train.�Ҹα��濹��.isnull(), '�Ҹα��濹��'] = 0
#     d_train.loc[d_train.�Ҹ�e��ԭ.isnull(), '�Ҹ�e��ԭ'] = 0
#     d_train.loc[d_train.�Ҹ�e����.isnull(), '�Ҹ�e����'] = 0
#     d_train.loc[d_train.�Ҹκ��Ŀ���.isnull(), '�Ҹκ��Ŀ���'] = 0

    d_train.drop(['�Ҹα��濹ԭ'],axis=1,inplace=True)
    d_train.drop(['�Ҹα��濹��'],axis=1,inplace=True)
    d_train.drop(['�Ҹ�e��ԭ'],axis=1,inplace=True)
    d_train.drop(['�Ҹ�e����'],axis=1,inplace=True)
    d_train.drop(['�Ҹκ��Ŀ���'],axis=1,inplace=True)
    
    d_train.drop(['id'],axis=1,inplace=True)
    
    d_train
    if num==1:
        train_y = d_train['Ѫ��'].as_matrix()
        d_train.drop(['Ѫ��'],axis=1,inplace=True)
        print(d_train.columns)
        d_train.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34']
        print(d_train.columns)
        train_x = d_train.as_matrix()
        return train_x, train_y
    d_train.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34']
    return d_train.as_matrix()

train_x, train_y = process_data(data_train, 1)
# test_x = process_data(data_test, 2)

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.2,random_state=0)

# from sklearn.ensemble import RandomForestRegressor
# from sklearn import grid_search
# rf=RandomForestRegressor(random_state=0,n_estimators=200,n_jobs=-1)
# rf.fit(x_train, y_train)
# print(rf.score(x_test,y_test))
# 0.13841243129
# 0.148252412959


from sklearn import grid_search
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
            "loss":['ls'],
            "learning_rate":[0.01],
            "n_estimators":[600],
            "max_features":[25,30], 
            "max_depth":[8], 
#               "min_samples_split": [200], 
#               "min_samples_leaf":[50], 
#             "subsample":[0.7], 
#               "min_impurity_split":[1e-7]
#             "warm_start":[True]
              }
# �����Ż�˳��n_estimators��max_depth��min_samples_split��min_samples_split��min_samples_leaf��max_features��subsample��learning_rate��n_estimators
gdbt=GradientBoostingRegressor(
                loss='lad', ##��ʧ�����ع�Ĭ��Ϊls����С���˷�����lad������ֵ���huber�����߽�ϣ���quantile����λ���������Ĭ��deviance  deviance���и�������ķ����ƫ��
                learning_rate=0.001, # Ĭ��0.1ѧϰ��
                n_estimators=6000, # ����ѵ���������ع�����������ѧϰ����
                max_depth=6, # Ĭ��ֵ3��������
                subsample=0.7, # �Ӳ����ʣ�Ϊ1��ʾȫ�����Ƽ�0.5-0.8Ĭ��1
                criterion='friedman_mse', # �жϽڵ��Ƿ�������Ѳ��õļ��㷽��
                min_samples_split=200, # �����ӽڵ�����Ҫ����С������������������ٷֱ�
                min_samples_leaf=50, # Ҷ�ӽڵ�����Ҫ����С������������������ٷֱ�
#                 min_weight_fraction_leaf=0.,
                min_impurity_split=1e-7, # ֹͣ����Ҷ�ӽڵ�ķ�ֵ
                init=None,
                random_state=0, # ������ӣ��������֣�û������ʲô 
#                 max_features=50, # Ѱ����ѷָ��Ҫ���ǵ���������auto��Noneȫѡ/sqrt����/log2����/int�Զ������/float�ٷֱ�
                verbose=0,
                max_leaf_nodes=None, # Ҷ�ӽڵ�ĸ�����None��������
                warm_start=False, # True��ǰ�����������ѵ���������������ѵ�������� FalseĬ��ֵ��������ѵ��
                presort='auto')  
# ���������Ͻ�����֤n_jobs=ģ�Ͳ��ж�1��cv=10������֤�Ĳ��Լ�Ϊ10�ۣ�verbose=10��ʾǰ10����ģ�ʹ�ӡ��־
gdbt = grid_search.GridSearchCV(estimator=gdbt, param_grid=param_grid, n_jobs=1, cv=5, verbose=10, scoring='neg_mean_squared_error')
gdbt.fit(train_x,train_y)

print(gdbt.best_estimator_)
print(gdbt.grid_scores_)
print(gdbt.best_params_)
print(gdbt.best_score_)

y_pred = gdbt.predict(test_x)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_y, y_pred)
print("MSE: %.4f" % mse)
# compute test set deviance
test_score = np.zeros((600,), dtype=np.float64)
for i, y_pred in enumerate(gdbt.staged_predict(test_x)):
    test_score[i] = gdbt.loss_(test_y, y_pred)
 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(600) + 1, gdbt.train_score_, 'b-',label='Training Set Deviance')
plt.plot(np.arange(600) + 1, test_score, 'r-',label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
 
###############################################################################
# Plot feature importance
feature_importance = gdbt.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


