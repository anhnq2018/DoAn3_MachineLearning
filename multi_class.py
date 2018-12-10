# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
import numpy as np
import math
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('You need to install matplotlib for plot_example.py.')

# load or create your dataset
print('Load data...')
df = pd.read_csv('winequality-white.csv',sep=';')
df_train, df_test = train_test_split(df, test_size=0.2)

y_train = df_train['quality'].values
y_test = df_test['quality'].values
X_train = df_train.drop('quality', axis=1).values
X_test = df_test.drop('quality', axis=1).values

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    #'boosting_type': 'cegb',
    'objective': 'regression',
    'num_leaves': 25,
    #'num_class' : 11,
    'feature_fraction': 0.5,
    'learning_rate': 0.005,
    'metric': 'l1',
    #'metric': 'multi_error',
    'bagging_fraction': 0.5,
    'bagging_freq': 10,
    #'max_bin': 150,
    'cegb_tradeoff': 0.01,
    #'cegb_penalty_split': 4,
    'is_unbalance': True,
    'verbose': 0,
}

evals_result = {}  # to record eval results for plotting
header = list(df.columns.values)
print(header[:-1])
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train, lgb_test],
                #early_stopping_rounds=100,
                #feature_name=['f' + str(i + 1) for i in range(11)],
                feature_name=header[:-1],
                evals_result=evals_result,
                verbose_eval=10)
#bst = lgb.Booster(model_file='model.txt')

print('Starting predicting...')
# predict
start = time.time()
y_pred = gbm.predict(X_train)
end = time.time()
# eval
#print(len(y_pred))
#print(y_test.shape)
#y_pred = np.where(y_pred > 0.5,1,0)
#y_pred = y_pred.argmax(axis=1)
#y_pred = [math.floor(float(x)) for x in y_pred]
#print(y_test.shape)
#print(y_pred)
print('Accuracy :',mean_squared_error(y_train, y_pred) ** 0.5)
print('time run : ' , end - start)

print('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='l1')
plt.show()

#print('Plot feature importances...')
#ax = lgb.plot_importance(gbm, max_num_features=10)
#plt.show()

print('Plot 84th tree with graphviz...')
graph = lgb.create_tree_digraph(gbm, tree_index=83, name='Tree84')
graph.render(view=True)
