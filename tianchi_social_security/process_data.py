#! usr/bin/python
#coding=utf-8
'''
Created on 2017年7月13日

@author: lianglong
'''

import pandas as pd
import numpy as np
 
df_train = pd.read_csv("data/df_train.csv", low_memory=False)
# df_train.info()
  
df_test = pd.read_csv("data/df_test.csv", low_memory=False)
# df_test.info()
 
df_id_train = pd.read_csv("data/df_id_train.csv", header=None)
# df_id_train.info()
 
df_id_test = pd.read_csv("data/df_id_test.csv", header=None)
# df_id_test.info()

# print(df_train.describe())
# 
# df_fee_detail = pd.read_csv("data/fee_detail.csv", low_memory=False)
# df_fee_detail.info()
# print(df_fee_detail.shape)

# print(df_train.head()['顺序号'])
# print(df_train.tail().icol(0))

# print(df_test.shape)
# print(df_train.shape)
# print(df_id_train.head())
# print(df_id_test.head())



# 处理数据流程
# 1. 先分析训练数据结构，补充缺失值和保留有效列，然后保留有效训练数据2w条，并保存到新文件中
# 2. 把训练数据放入分类模型中，使用网格参数，对数据进行训练


def process_data(df_train, df_id_train, df_test, df_id_test):
    # 标记数据添加列头
    df_id_train.columns = ["个人编码", "1"]
    
    # 提取次数特征
    df_count = df_train.groupby("个人编码", as_index=False)['医院编码'].count()
    df_count.columns = ["个人编码", "次数"]
    
    # 提取累加特征
    df_train = df_train.groupby("个人编码", as_index=False).sum()
    
    # 合并次数、累加、标记数据
    df_train = pd.merge(df_count, df_train, how='inner', on=None, left_on="个人编码", right_on="个人编码")
    result = pd.merge(df_id_train, df_train, how='inner', on=None, left_on="个人编码", right_on="个人编码")
    result.info()
    # 写入到csv文件中
    result.to_csv('data/train.csv', index=False, encoding="utf-8")  

    # 测试数据做相同处理
    df_id_test.columns = ["个人编码"]
    df_count = df_test.groupby("个人编码", as_index=False)['医院编码'].count()
    df_count.columns = ["个人编码", "次数"]
    df_test = df_test.groupby("个人编码", as_index=False).sum()
    df_test = pd.merge(df_count, df_test, how='inner', on=None, left_on="个人编码", right_on="个人编码")
    result = pd.merge(df_id_test, df_test, how='inner', on=None, left_on="个人编码", right_on="个人编码")
    result.to_csv('data/test.csv', index=False, encoding="utf-8")  
    
process_data(df_train, df_id_train, df_test, df_id_test)

