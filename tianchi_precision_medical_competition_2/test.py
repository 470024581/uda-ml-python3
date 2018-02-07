import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

data_train = pd.read_csv('data/f_train_20180204.csv', encoding='gbk')

# data_train.drop(['id'],axis=1,inplace=True)
data_train.drop(['RBP4'],axis=1,inplace=True)
data_train.drop(['SNP10'],axis=1,inplace=True)
data_train.drop(['SNP21'],axis=1,inplace=True)
data_train.drop(['SNP3'],axis=1,inplace=True)
data_train.drop(['产次'],axis=1,inplace=True)
data_train.drop(['SNP53'],axis=1,inplace=True)
data_train.drop(['SNP5'],axis=1,inplace=True)
data_train.drop(['SNP30'],axis=1,inplace=True)
data_train.drop(['SNP8'],axis=1,inplace=True)
data_train.drop(['SNP9'],axis=1,inplace=True)
data_train.drop(['SNP4'],axis=1,inplace=True)
data_train.drop(['BMI分类'],axis=1,inplace=True)
data_train.drop(['SNP33'],axis=1,inplace=True)
data_train.drop(['SNP24'],axis=1,inplace=True)
data_train.drop(['SNP16'],axis=1,inplace=True)
data_train.drop(['SNP35'],axis=1,inplace=True)
data_train.drop(['SNP2'],axis=1,inplace=True)
data_train.drop(['SNP25'],axis=1,inplace=True)
data_train.drop(['SNP15'],axis=1,inplace=True)
data_train.drop(['SNP14'],axis=1,inplace=True)
data_train.drop(['分娩时'],axis=1,inplace=True)

data_train.info()

# data_train.fillna(data_train.median(axis=0),inplace=True)

# 补充中位数的列：
## 需要预测补全的列 'SNP1',
# 缺失一半：SNP22,SNP23,SNP54,SNP55,ACEID,DM家族史
# 缺失五分之一：孕次,身高,收缩压,舒张压,糖筛孕周,AST
# 缺失十分之一：wbc,ALT,Cr,BUN


def pred_data(train, rf, unknown_columns):
    y=train[:,0]  
    x=train[:,1:]  
    rf.fit(x,y)  
    print(unknown_columns)
    predict=rf.predict(unknown_columns[:,1:])  
    return predict

#预测年纪并补全  
# def set_missing_SNP22(data):  
#     data=data[['SNP22','SNP23','SNP54','SNP55','ACEID','DM家族史',
#                 'SNP1','SNP6','SNP7','SNP11','SNP12','SNP13','SNP17','SNP18','SNP19','SNP20','年龄','VAR00007','CHO','TG','HDLC','LDLC','ApoA1','ApoB','Lpa'
#                 ,'hsCRP','SNP26','SNP27','SNP28','SNP29','SNP31','SNP32','SNP34','SNP36','SNP37','SNP38','SNP39','SNP40','SNP41','SNP42','SNP43','SNP44','SNP46'
#                 ,'SNP47','SNP48','SNP49','SNP50','SNP51','SNP52','SNP54','SNP55','label'
#                 ,'wbc','ALT','Cr','BUN'
#                 ,'孕次','身高','收缩压','舒张压','糖筛孕周','AST']]  
#     train=data[[#'SNP22','SNP23','SNP54','SNP55','ACEID','DM家族史',
#                 'SNP1','SNP6','SNP7','SNP11','SNP12','SNP13','SNP17','SNP18','SNP19','SNP20','年龄','VAR00007','CHO','TG','HDLC','LDLC','ApoA1','ApoB','Lpa'
#                 ,'hsCRP','SNP26','SNP27','SNP28','SNP29','SNP31','SNP32','SNP34','SNP36','SNP37','SNP38','SNP39','SNP40','SNP41','SNP42','SNP43','SNP44','SNP46'
#                 ,'SNP47','SNP48','SNP49','SNP50','SNP51','SNP52','SNP54','SNP55','label'
#                 ,'wbc','ALT','Cr','BUN'
#                 ,'孕次','身高','收缩压','舒张压','糖筛孕周','AST']]  
#     train.fillna(train.median(axis=0),inplace=True)
#     unknown_column=data[data.SNP22.isnull()].as_matrix()  
#     unknown_SNP23=data[data.SNP23.isnull()].as_matrix()  
#     unknown_SNP54=data[data.SNP54.isnull()].as_matrix()  
#     unknown_SNP55=data[data.SNP55.isnull()].as_matrix()  
#     unknown_ACEID=data[data.ACEID.isnull()].as_matrix()  
#     unknown_column=data[data.DM家族史.isnull()].as_matrix()  
#     rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
#     pred_SNP22 = pred_data(train.as_matrix(), rf, unknown_column)
#     data.loc[data.SNP22.isnull(),'SNP22']=predict
#     return data  
# 
# 
# data = set_missing_SNP22(data_train)  



def set_missing(data):  
    data_SNP22 = data[['SNP22','id']]
    data_SNP23 = data[['SNP23','id']]
    data_SNP54 = data[['SNP54','id']]
    data_SNP55 = data[['SNP55','id']]
    data_ACEID = data[['ACEID','id']]
    data_DM家族史 = data[['DM家族史','id']]
    
    data.drop(['SNP22','SNP23','SNP54','SNP55','ACEID','DM家族史'],axis=1,inplace=True)
    data.fillna(data.median(axis=0),inplace=True)
    
    train = pd.merge(data_DM家族史, data, on='id')
#     train.drop(['id'],axis=1,inplace=True)
    known_column=train[train.DM家族史.notnull()]
    known_column = known_column.as_matrix()  
    DM家族史=train[train.DM家族史.isnull()].as_matrix()  
    y=known_column[:,0]  
    x=known_column[:,1:]  
    rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
    rf.fit(x,y)  
    predictDM家族史=rf.predict(DM家族史[:,1:])  
    train.loc[train.DM家族史.isnull(),'DM家族史']=predictDM家族史  
    
    
    train = pd.merge(data_ACEID, train, on='id')
    known_column=train[train.ACEID.notnull()]
    known_column = known_column.as_matrix()  
    unknown_column=train[train.ACEID.isnull()].as_matrix()  
    y=known_column[:,0]  
    x=known_column[:,1:]  
    rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
    rf.fit(x,y)  
    predict=rf.predict(unknown_column[:,1:])  
    train.loc[train.ACEID.isnull(),'ACEID']=predict  
    
    
    train = pd.merge(data_SNP55, train, on='id')
    known_column=train[train.SNP55.notnull()]
    known_column = known_column.as_matrix()  
    unknown_column=train[train.SNP55.isnull()].as_matrix()  
    y=known_column[:,0]  
    x=known_column[:,1:]  
    rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
    rf.fit(x,y)  
    predict=rf.predict(unknown_column[:,1:])  
    train.loc[train.SNP55.isnull(),'SNP55']=predict  
    
    
    train = pd.merge(data_SNP54, train, on='id')
    known_column=train[train.SNP54.notnull()]
    known_column = known_column.as_matrix()  
    unknown_column=train[train.SNP54.isnull()].as_matrix()  
    y=known_column[:,0]  
    x=known_column[:,1:]  
    rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
    rf.fit(x,y)  
    predict=rf.predict(unknown_column[:,1:])  
    train.loc[train.SNP54.isnull(),'SNP54']=predict  
    
    
    train = pd.merge(data_SNP23, train, on='id')
    known_column=train[train.SNP23.notnull()]
    known_column = known_column.as_matrix()  
    unknown_column=train[train.SNP23.isnull()].as_matrix()  
    y=known_column[:,0]  
    x=known_column[:,1:]  
    rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
    rf.fit(x,y)  
    predict=rf.predict(unknown_column[:,1:])  
    train.loc[train.SNP23.isnull(),'SNP23']=predict  
    
    
    train = pd.merge(data_SNP22, train, on='id')
    known_column=train[train.SNP22.notnull()]
    known_column = known_column.as_matrix()  
    unknown_column=train[train.SNP22.isnull()].as_matrix()  
    y=known_column[:,0]  
    x=known_column[:,1:]  
    rf=RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)  
    rf.fit(x,y)  
    predict=rf.predict(unknown_column[:,1:])  
    train.loc[train.SNP22.isnull(),'SNP22']=predict  
    
    return train

data = set_missing(data_train)  
data.info()

