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


# create dataset for lightgbm
lgb_train = lgb.Dataset(trn_X, trn_y)
lgb_eval = lgb.Dataset(val_X, val_y,reference=lgb_train)#



# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
#    'metric': {'mse'},
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


y_true=np.array(val_y)

#print( myeval(gbm.predict(trn_X, num_iteration=gbm.best_iteration), lgb_train))
print( myeval(gbm.predict(val_X, num_iteration=gbm.best_iteration), lgb_eval))
print( myeval(gbm.predict(val_X), lgb_eval))