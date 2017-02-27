# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:28:19 2017

@author: charles
"""
import pandas as pd
import pickle as pkl

f=open('../sample/comb.pkl','rb')
comb=pkl.load(f)
f.close()


trn_sample_idx=(comb.date>=pd.Timestamp("2016-09-15"))&(comb.date<=pd.Timestamp("2016-10-31"))
val_sample_idx=(comb.date>=pd.Timestamp("2016-09-01"))&(comb.date<=pd.Timestamp("2016-09-14"))
tst_sample_idx=comb.date>=pd.Timestamp("2016-11-01")


X=comb.drop(['pay_cnt','date'], axis=1)
print('ready to get_dummies')
X=pd.get_dummies(X)
print('get_dummies done')
y=comb['pay_cnt']


trn_y=y[trn_sample_idx]
val_y=y[val_sample_idx]
trn_y.reset_index(drop=True,inplace=True)
val_y.reset_index(drop=True,inplace=True)

trn_X=X[trn_sample_idx]
val_X=X[val_sample_idx]
tst_X=X[tst_sample_idx]
trn_X.reset_index(drop=True,inplace=True)
val_X.reset_index(drop=True,inplace=True)
tst_X.reset_index(drop=True,inplace=True)




f=open('../sample/trn_y','wb')
pkl.dump(trn_y,f)
f.close()
f=open('../sample/val_y','wb')
pkl.dump(val_y,f)
f.close()
f=open('../sample/trn_X','wb')
pkl.dump(trn_X,f)
f.close()
f=open('../sample/val_X','wb')
pkl.dump(val_X,f)
f.close()
f=open('../sample/tst_X','wb')
pkl.dump(tst_X,f)
f.close()