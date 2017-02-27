# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:30:07 2017

@author: charles
"""

import numpy as np
import pickle as pkl

f = open('../sample/trn_y','rb')
trn_y=pkl.load(f)
f.close()
f=open('../sample/val_y','rb')
val_y=pkl.load(f)
f.close()
f=open('../sample/trn_X','rb')
trn_X=pkl.load(f)
f.close()
f=open('../sample/val_X','rb')
val_X=pkl.load(f)
f.close()
f=open('../sample/tst_X','rb')
tst_X=pkl.load(f)
f.close()


def myeval(preds, train_data):
    labels = train_data.get_label()
    value=np.mean(abs((abs(preds)-labels)/(abs(preds)+labels)))
    return 'loss', value, False




import lightgbm as lgb


"""
y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


"""


# create dataset for lightgbm
lgb_train = lgb.Dataset(trn_X, trn_y)
lgb_eval = lgb.Dataset(val_X, val_y,reference=lgb_train)#


# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=50,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
##---
"""
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mse'},
    'num_leaves': 31,
    'learning_rate': 0.2,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                valid_names="ValidSet",
                fobj=None, 
#                feval=None,
                feval=myeval, 
                early_stopping_rounds=30
                )
"""
###---
#print('Save model...')
## save model to file
#gbm.save_model('model.txt')
#
#print('Start predicting...')
## predict
#y_pred = gbm.predict(val_X, num_iteration=gbm.best_iteration)
## eval
#
y_true=np.array(val_y)

#print( myeval(gbm.predict(trn_X, num_iteration=gbm.best_iteration), lgb_train))
#print( myeval(gbm.predict(val_X, num_iteration=gbm.best_iteration), lgb_eval))
print( myeval(gbm.predict(val_X), lgb_eval))
from sklearn.metrics import mean_squared_error
print('The rmse of prediction is:', mean_squared_error(val_y,gbm.predict(val_X)) ** 0.5)