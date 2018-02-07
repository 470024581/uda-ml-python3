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

data_train = pd.read_csv("data/d_train.csv", low_memory=False)
data_test = pd.read_csv("data/d_test_A_20180102.csv", low_memory=False)

# d_train.info()
# print(d_train.head(590))
# print(d_train.describe())
# print(d_train.shape)
# print(d_train['性别'])
# print(d_train['血糖'])

# 数据预处理
# 删除无用列
def process_data(d_train, num):
    d_train.drop(['体检日期'],axis=1,inplace=True)
    # 删除一行id为580的数据，因为性别为？？
    if num==1:
        d_train.drop(d_train[(d_train.id==580)].index,inplace=True)
#         d_train.drop(d_train[(d_train.血糖>=20)].index,inplace=True)
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
#         d_train.drop(d_train[(d_train.血糖>=20)].index,inplace=True)
        d_train.drop(d_train[(d_train.血糖<=min)].index,inplace=True)
        d_train.drop(d_train[(d_train.血糖>=max)].index,inplace=True)
        
    # 转换列名
    d_train['嗜碱细胞百分比'] = d_train['嗜碱细胞%']
    d_train.drop(['嗜碱细胞%'],axis=1,inplace=True)
    d_train['嗜酸细胞百分比'] = d_train['嗜酸细胞%']
    d_train.drop(['嗜酸细胞%'],axis=1,inplace=True)
    d_train['单核细胞百分比'] = d_train['单核细胞%']
    d_train.drop(['单核细胞%'],axis=1,inplace=True)
    d_train['淋巴细胞百分比'] = d_train['淋巴细胞%']
    d_train.drop(['淋巴细胞%'],axis=1,inplace=True)
    d_train['中性粒细胞百分比'] = d_train['中性粒细胞%']
    d_train.drop(['中性粒细胞%'],axis=1,inplace=True)
    
    d_train['_天门冬氨酸氨基转换酶'] = d_train['*天门冬氨酸氨基转换酶']
    d_train.drop(['*天门冬氨酸氨基转换酶'],axis=1,inplace=True)
    d_train['_丙氨酸氨基转换酶'] = d_train['*丙氨酸氨基转换酶']
    d_train.drop(['*丙氨酸氨基转换酶'],axis=1,inplace=True)
    d_train['_碱性磷酸酶'] = d_train['*碱性磷酸酶']
    d_train.drop(['*碱性磷酸酶'],axis=1,inplace=True)
    d_train['_r_谷氨酰基转换酶'] = d_train['*r-谷氨酰基转换酶']
    d_train.drop(['*r-谷氨酰基转换酶'],axis=1,inplace=True)
    d_train['_总蛋白'] = d_train['*总蛋白']
    d_train.drop(['*总蛋白'],axis=1,inplace=True)
    d_train['_球蛋白'] = d_train['*球蛋白']
    d_train.drop(['*球蛋白'],axis=1,inplace=True)
    # 枚举类型转换成数值类型
    sex_mapping = {'男':1,'女':2}
    d_train['性别'] = d_train['性别'].map(sex_mapping)
    # 缺失少量数据用中位数补充
    # 缺失14和21的值，一共17列
    d_train.loc[d_train.白细胞计数.isnull(), '白细胞计数'] = d_train['白细胞计数'].median()
    d_train.loc[d_train.红细胞计数.isnull(), '红细胞计数'] = d_train['红细胞计数'].median()
    d_train.loc[d_train.血红蛋白.isnull(), '血红蛋白'] = d_train['血红蛋白'].median()
    d_train.loc[d_train.红细胞压积.isnull(), '红细胞压积'] = d_train['红细胞压积'].median()
    d_train.loc[d_train.红细胞平均体积.isnull(), '红细胞平均体积'] = d_train['红细胞平均体积'].median()
    d_train.loc[d_train.红细胞平均血红蛋白量.isnull(), '红细胞平均血红蛋白量'] = d_train['红细胞平均血红蛋白量'].median()
    d_train.loc[d_train.红细胞平均血红蛋白浓度.isnull(), '红细胞平均血红蛋白浓度'] = d_train['红细胞平均血红蛋白浓度'].median()
    d_train.loc[d_train.红细胞体积分布宽度.isnull(), '红细胞体积分布宽度'] = d_train['红细胞体积分布宽度'].median()
    d_train.loc[d_train.血小板计数.isnull(), '血小板计数'] = d_train['血小板计数'].median() 
    d_train.loc[d_train.血小板平均体积.isnull(), '血小板平均体积'] = d_train['血小板平均体积'].median()
    d_train.loc[d_train.血小板体积分布宽度.isnull(), '血小板体积分布宽度'] = d_train['血小板体积分布宽度'].median()
    d_train.loc[d_train.血小板比积.isnull(), '血小板比积'] = d_train['血小板比积'].median()
    d_train.loc[d_train.中性粒细胞百分比.isnull(), '中性粒细胞百分比'] = d_train['中性粒细胞百分比'].median()
    d_train.loc[d_train.淋巴细胞百分比.isnull(), '淋巴细胞百分比'] = d_train['淋巴细胞百分比'].median() # 26
    d_train.loc[d_train.单核细胞百分比.isnull(), '单核细胞百分比'] = d_train['单核细胞百分比'].median() # 25
    d_train.loc[d_train.嗜酸细胞百分比.isnull(), '嗜酸细胞百分比'] = d_train['嗜酸细胞百分比'].median() # 24
    d_train.loc[d_train.嗜碱细胞百分比.isnull(), '嗜碱细胞百分比'] = d_train['嗜碱细胞百分比'].median() # 23
    
#     d_train.drop(['红细胞压积'],axis=1,inplace=True)
#     d_train.drop(['红细胞平均血红蛋白量'],axis=1,inplace=True)
#     d_train.drop(['血小板计数'],axis=1,inplace=True)
#     d_train.drop(['血小板平均体积'],axis=1,inplace=True)
#     d_train.drop(['血小板体积分布宽度'],axis=1,inplace=True)
#     d_train.drop(['血小板比积'],axis=1,inplace=True)
#     d_train.drop(['淋巴细胞百分比'],axis=1,inplace=True)
#     d_train.drop(['单核细胞百分比'],axis=1,inplace=True)
#     d_train.drop(['嗜酸细胞百分比'],axis=1,inplace=True)
#     d_train.drop(['嗜碱细胞百分比'],axis=1,inplace=True)
    
    # 缺失1219、1221、1378的值，分别4、8、3列，一共15列
    # from scipy.stats import mode
    #　mode(d_train['甘油三酯'])　众数
    
    d_train.loc[d_train.甘油三酯.isnull(), '甘油三酯'] = d_train['甘油三酯'].median()
    d_train.loc[d_train.总胆固醇.isnull(), '总胆固醇'] = d_train['总胆固醇'].median()
    d_train.loc[d_train.高密度脂蛋白胆固醇.isnull(), '高密度脂蛋白胆固醇'] = d_train['高密度脂蛋白胆固醇'].median()
    d_train.loc[d_train.低密度脂蛋白胆固醇.isnull(), '低密度脂蛋白胆固醇'] = d_train['低密度脂蛋白胆固醇'].median()
    d_train.loc[d_train._天门冬氨酸氨基转换酶.isnull(), '_天门冬氨酸氨基转换酶'] = d_train['_天门冬氨酸氨基转换酶'].median()
    d_train.loc[d_train._丙氨酸氨基转换酶.isnull(), '_丙氨酸氨基转换酶'] = d_train['_丙氨酸氨基转换酶'].median()
    d_train.loc[d_train._碱性磷酸酶.isnull(), '_碱性磷酸酶'] = d_train['_碱性磷酸酶'].median()
    d_train.loc[d_train._r_谷氨酰基转换酶.isnull(), '_r_谷氨酰基转换酶'] = d_train['_r_谷氨酰基转换酶'].median()
#     d_train.loc[d_train._总蛋白.isnull(), '_总蛋白'] = d_train['_总蛋白'].median() 33
    d_train.loc[d_train.白蛋白.isnull(), '白蛋白'] = d_train['白蛋白'].median()
#     d_train.loc[d_train._球蛋白.isnull(), '_球蛋白'] = d_train['_球蛋白'].median() 34
    d_train.loc[d_train.白球比例.isnull(), '白球比例'] = d_train['白球比例'].median()
    d_train.loc[d_train.尿素.isnull(), '尿素'] = d_train['尿素'].median()
    d_train.loc[d_train.肌酐.isnull(), '肌酐'] = d_train['肌酐'].median()
    d_train.loc[d_train.尿酸.isnull(), '尿酸'] = d_train['尿酸'].median()
    
    d_train.drop(['_总蛋白'],axis=1,inplace=True)
    d_train.drop(['_球蛋白'],axis=1,inplace=True)
    
    # 缺失大量数据（若数据不重要直接剔除，数据比较重要则要小心处理）大量缺失数据不适合中位数填充，
#     d_train.loc[d_train.乙肝表面抗原.isnull(), '乙肝表面抗原'] = 0
#     d_train.loc[d_train.乙肝表面抗体.isnull(), '乙肝表面抗体'] = 0
#     d_train.loc[d_train.乙肝e抗原.isnull(), '乙肝e抗原'] = 0
#     d_train.loc[d_train.乙肝e抗体.isnull(), '乙肝e抗体'] = 0
#     d_train.loc[d_train.乙肝核心抗体.isnull(), '乙肝核心抗体'] = 0

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
# 参数优化顺序n_estimators、max_depth和min_samples_split、min_samples_split和min_samples_leaf、max_features、subsample、learning_rate和n_estimators
gdbt=GradientBoostingRegressor(
                loss='lad', ##损失函数回归默认为ls（最小二乘法）、lad（绝对值差）、huber（二者结合）、quantile（分位数差），分类默认deviance  deviance具有概率输出的分类的偏差
                learning_rate=0.001, # 默认0.1学习率
                n_estimators=6000, # 迭代训练次数，回归树个数（弱学习器）
                max_depth=6, # 默认值3最大树深度
                subsample=0.7, # 子采样率，为1表示全部，推荐0.5-0.8默认1
                criterion='friedman_mse', # 判断节点是否继续分裂采用的计算方法
                min_samples_split=200, # 生成子节点所需要的最小样本数，浮点数代表百分比
                min_samples_leaf=50, # 叶子节点所需要的最小样本数，浮点数代表百分比
#                 min_weight_fraction_leaf=0.,
                min_impurity_split=1e-7, # 停止分裂叶子节点的阀值
                init=None,
                random_state=0, # 随机种子，方便重现，没有明白什么 
#                 max_features=50, # 寻找最佳分割点要考虑的特征数量auto和None全选/sqrt开方/log2对数/int自定义个数/float百分比
                verbose=0,
                max_leaf_nodes=None, # 叶子节点的个数，None不限数量
                warm_start=False, # True在前面基础上增量训练（重设参数减少训练次数） False默认值擦除重新训练
                presort='auto')  
# 网格参数配合交叉验证n_jobs=模型并行度1，cv=10交叉验证的测试集为10折，verbose=10表示前10个子模型打印日志
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


