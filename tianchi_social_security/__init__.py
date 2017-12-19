
# -*- coding: utf-8 -*-
#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
# 导入自动美化matplotlib图形
import seaborn as sns
import pandas as pd

x = np.linspace(0,4*3.1415,100)
y = np.sin(x)

data = [1,2,3,4]

plt.hist(data, bins=20)

# plt.figure(figsize=(8,4))
# plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
# plt.legend()
# plt.show()


df_fee_detail = pd.read_csv("data/fee_detail.csv", low_memory=False)
df_fee_count = df_fee_detail.groupby(["顺序号","医院编码"], as_index=False)['三目统计项目'].count()
df_fee_count.info()
print(df_fee_count)
print(df_fee_count.loc[df_fee_count.三目统计项目 >= 2, '三目统计项目'])

# [2192987 rows x 2 columns]

# df_fee_detail.info()
# 
# print(df_fee_detail.head())
# print("==========================================")
# df_test = pd.read_csv("data/df_test.csv", low_memory=False)
# df_test.info()