#! usr/bin/python
#coding=utf-8
'''
Created on 2017年7月24日

@author: lianglong
'''

import pandas as pd
import numpy as np

train_data = pd.read_csv("data2/train.csv", low_memory=False)
test_data = pd.read_csv("data2/test.csv", low_memory=False)
id_data = test_data['个人编码']
def process_data(train_data):
#     train_data.info()
    
    # 删除没有数据的列
    train_data.drop(['一次性医用材料拒付金额','农民工医疗救助计算金额','残疾军人医疗补助基金支付金额','拒付原因编码'],axis=1,inplace=True)
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
        train_y = train_data["标记"].as_matrix()
#         train_data.drop(["个人编码","标记",'操作时间','申报受理时间','出院诊断病种名称','住院开始时间','住院终止时间','交易时间','顺序号'],axis=1,inplace=True)
        train_data.drop(["个人编码","标记"],axis=1,inplace=True)
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

from sklearn import metrics
# one-hot
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# 归一化处理
from sklearn.preprocessing import StandardScaler

train_y=LabelEncoder().fit(train_y).transform(train_y)  
train_y=OneHotEncoder(sparse=False).fit(train_y.reshape(-1,1)).transform(train_y.reshape(-1,1))  

# from sklearn.model_selection import train_test_split  
# x_tr,x_te,y_tr,y_te=train_test_split(train_x,train_y,test_size=0.2,random_state=0)  

# data_train = pd.read_csv('train.csv')
# data_test = pd.read_csv('test.csv')
# train_x, train_y, test_x = process_data(data_train, data_test)

import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros(shape=[1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, Weight) + bias
#     wx_plus_b = tf.nn.bias_add(wx_plus_b, bias)
    y = tf.nn.dropout(wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = y
    else:
        outputs = activation_function(y)
    return outputs

keep_prob = tf.placeholder(tf.float32)


def next_batch(self, batch_size):  
    start = self._index_in_epoch  
    self._index_in_epoch += batch_size  
    if self._index_in_epoch > self._num_examples:  
        # Finished epoch  
        self._epochs_completed += 1  
        # Shuffle the data  
        perm = np.arange(self._num_examples)  
        np.random.shuffle(perm)  
        self._images = self._images[perm]  
        self._labels = self._labels[perm]  
        # Start next epoch  
        start = 0  
        self._index_in_epoch = batch_size  
        assert batch_size <= self._num_examples  
    end = self._index_in_epoch  
    return self._images[start:end], self._labels[start:end]  


    # 定义变量
def train_network():
    
    x = tf.placeholder(tf.float32, [None, 64])
    y = tf.placeholder(tf.float32, [None, 2])
     
    l1 = add_layer(x, 64, 32, activation_function=tf.nn.tanh)
    l2 = add_layer(l1, 32, 16, activation_function=tf.nn.tanh)
    l3 = add_layer(l2, 16, 8, activation_function=tf.nn.tanh)
    l4 = add_layer(l3, 8, 4, activation_function=tf.nn.tanh)
    l7 = add_layer(l4, 4, 2, activation_function=tf.nn.softmax)
    prediction = add_layer(l7, 2, 2, activation_function=None)
     
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1)), tf.float32))
    train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)
    
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    
    for i in range(1000):
        session.run(train_step, feed_dict={x: train_x, y: train_y, keep_prob: 0.5})
        if i % 100 == 0:
            result = session.run(accuracy, feed_dict={x: train_x, y: train_y, keep_prob: 0.5})
            print("accuracy===", result)
#             pred = session.run(prediction, feed_dict={x: test_x, y: test_y, keep_prob: 0.5})
#             print(pred)
    
train_network()

