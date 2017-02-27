# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:30:07 2017

@author: charles
"""

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
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



X_train=trn_X
y_train=trn_y
X_test=val_X
y_test=val_y

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)



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
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

#print('Save model...')
## save model to file
#gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
