#! usr/bin/python
#coding=utf-8
"""
Created on 2017年7月13日

@author: lianglong
"""

import pandas as pd
import numpy as np
# 加载数据集
df_fee_detail = pd.read_csv("data/fee_detail.csv", low_memory=False)
df_train = pd.read_csv("data/df_train.csv", low_memory=False)
df_test = pd.read_csv("data/df_test.csv", low_memory=False)
df_id_train = pd.read_csv("data/df_id_train.csv", header=None)
df_id_test = pd.read_csv("data/df_id_test.csv", header=None)

df_train[["个人编码"]] = df_train[["个人编码"]].astype(np.int64)
df_test[["个人编码"]] = df_test[["个人编码"]].astype(np.int64)
df_train[["医院编码"]] = df_train[["医院编码"]].astype(np.int64)
df_test[["医院编码"]] = df_test[["医院编码"]].astype(np.int64)



# 提取次数特征
df_fee_count = df_fee_detail.groupby(["顺序号"], as_index=False)["三目统计项目"].count()
df_train_count = df_train.groupby(["个人编码"], as_index=False)["医院编码"].count()
df_test_count = df_test.groupby(["个人编码"], as_index=False)["医院编码"].count()
df_fee_count.columns = ["顺序号", "医院附加明细数量"]
df_train_count.columns = ["个人编码", "个人医保数量"]
df_test_count.columns = ["个人编码", "个人医保数量"]

df_fee_detail = pd.merge(df_fee_detail, df_fee_count, how="inner", on=None, left_on="顺序号", right_on="顺序号")
df_train = pd.merge(df_train, df_fee_detail, how="inner", on=None, left_on="顺序号", right_on="顺序号")
df_test = pd.merge(df_test, df_fee_detail, how="inner", on=None, left_on="顺序号", right_on="顺序号")

# 合并数据集为训练数据和测试数据
df_train = df_train.groupby(["个人编码"], as_index=False).sum()
df_test = df_test.groupby(["个人编码"], as_index=False).sum()

df_train = pd.merge(df_train_count, df_train, how="inner", on=None, left_on="个人编码", right_on="个人编码")
df_test = pd.merge(df_test_count, df_test, how="inner", on=None, left_on="个人编码", right_on="个人编码")

# df_fee_detail.info()
# df_train.info()
# df_test.info()
# print("========================")

# 添加标记数据
df_id_train.columns = ["个人编码", "标记"]
df_id_test.columns = ["个人编码"]
df_train = pd.merge(df_id_train, df_train, how="inner", on=None, left_on="个人编码", right_on="个人编码")
df_test = pd.merge(df_id_test, df_test, how="inner", on=None, left_on="个人编码", right_on="个人编码")

df_train.info()
df_test.info()

df_train.to_csv("data2/train.csv", index=False, encoding="utf-8")
df_test.to_csv("data2/test.csv", index=False, encoding="utf-8")


