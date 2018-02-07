import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error



data_path = 'data/'

train = pd.read_csv(data_path+'d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv(data_path+'d_test_A_20180102.csv', encoding='utf8')

def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])

    data['�Ա�'] = data['�Ա�'].map({'��':1,'Ů':0})
    data['�������'] = (pd.to_datetime(data['�������']) - parse('2017-10-09')).dt.days
    # ȱʧֵ������λ��
    data.fillna(data.median(axis=0),inplace=True)

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    test_feat.info()
    test_feat.drop(['�Ҹα��濹ԭ'],axis=1,inplace=True)
    test_feat.drop(['�Ҹα��濹��'],axis=1,inplace=True)
    test_feat.drop(['�Ҹ�e��ԭ'],axis=1,inplace=True)
    test_feat.drop(['�Ҹ�e����'],axis=1,inplace=True)
    test_feat.drop(['�Ҹκ��Ŀ���'],axis=1,inplace=True)
    train_feat.drop(['�Ҹα��濹ԭ'],axis=1,inplace=True)
    train_feat.drop(['�Ҹα��濹��'],axis=1,inplace=True)
    train_feat.drop(['�Ҹ�e��ԭ'],axis=1,inplace=True)
    train_feat.drop(['�Ҹ�e����'],axis=1,inplace=True)
    train_feat.drop(['�Ҹκ��Ŀ���'],axis=1,inplace=True)
    test_feat.info()
    
    return train_feat,test_feat



train_feat,test_feat = make_feat(train,test)

predictors = [f for f in test_feat.columns if f not in ['Ѫ��']]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)

print('��ʼѵ��...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('��ʼCV 5��ѵ��...')
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('��{}��ѵ��...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['Ѫ��'],categorical_feature=['�Ա�'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['Ѫ��'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=6000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:,i] = gbm.predict(test_feat[predictors])
print('���µ÷֣�    {}'.format(mean_squared_error(train_feat['Ѫ��'],train_preds)*0.5))
print('CVѵ����ʱ{}��'.format(time.time() - t0))

submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None, index=False, float_format='%.4f')

