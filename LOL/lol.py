# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/9 15:52
# Filename: lol.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb


train_datapath = '../data/test/train.csv.zip'
test_datapath = '../data/test/test.csv.zip'


def read_data(train_datapath='data/test/train.csv.zip', test_datapath='test/test.csv.zip'):
    # read csv data
    train_df = pd.read_csv(train_datapath)
    test_df = pd.read_csv(test_datapath)

    # 删除id列
    # timecc列的值都为0，也进行删除
    train_df = train_df.drop(['id', 'timecc'], axis=1)
    test_df = test_df.drop(['id', 'timecc'], axis=1)

    # 归一化处理
    for col in train_df.columns[1:]:
        train_df[col] /= train_df[col].max()
        test_df[col] /= test_df[col].max()

    # 数据类型转换
    train_data = train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    train_label = train_df.iloc[:, 0].to_numpy(dtype=np.int64)
    test_data = test_df.iloc[:].to_numpy(dtype=np.float32)

    return train_data, train_label, test_data


if __name__ == '__main__':
    train_data, train_label, test_data = read_data(train_datapath, test_datapath)

    seed = 10086
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.1, random_state=seed)

    # use gpu
    model = xgb.XGBClassifier(n_estimators=150, random_state=seed, tree_method='gpu_hist', subsample=0.6870252525252525,
                              learning_rate=0.13394170170170172)
    model.fit(x_train, y_train)

    score = model.score(x_test, y_test)
    print('test score: ', score)
    cvs_score = cross_val_score(model, train_data, train_label, cv=10)
    print('train score: ', cvs_score.mean())

    # test score:  0.8563888888888889
    # train score:  0.854238888888889
    # submission: 0.84455(划分训练集)
    # submission: 0.8445(未划分训练集)
