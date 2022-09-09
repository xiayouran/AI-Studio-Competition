# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/9 15:04
# Filename: pubg.py
import paddle
import pandas as pd
import numpy as np


train_datapath = '../data/pubg/pubg_train.csv.zip'
test_datapath = '../data/pubg/pubg_test.csv.zip'


def read_data(train_datapath='data/pubg/pubg_train.csv.zip', test_datapath='data/pubg/pubg_test.csv.zip'):
    # read csv data
    train_df = pd.read_csv(train_datapath)
    test_df = pd.read_csv(test_datapath)

    # delete invalid field
    train_df = train_df.drop(['match_id', 'team_id', 'player_name', 'kill_distance_x_min', 'kill_distance_x_max', 'kill_distance_y_min', 'kill_distance_y_max'], axis=1)
    test_df = test_df.drop(['match_id', 'team_id', 'player_name', 'kill_distance_x_min', 'kill_distance_x_max', 'kill_distance_y_min', 'kill_distance_y_max'], axis=1)

    # fill nan to 0
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # 数值归一化
    for col in train_df.columns[:-1]:
        train_df[col] /= train_df[col].max()
        test_df[col] /= test_df[col].max()

    return train_df, test_df


class PUBGRegressor(paddle.nn.Layer):
    """数据量很大，建议尝试深层神经网络"""
    def __init__(self):
        super(PUBGRegressor, self).__init__()

        self.fc1 = paddle.nn.Linear(in_features=8, out_features=64)
        self.fc2 = paddle.nn.Linear(in_features=64, out_features=128)
        self.fc3 = paddle.nn.Linear(in_features=128, out_features=256)
        self.fc4 = paddle.nn.Linear(in_features=256, out_features=512)
        self.fc5 = paddle.nn.Linear(in_features=512, out_features=1024)
        self.fc6 = paddle.nn.Linear(in_features=1024, out_features=2048)
        self.fc7 = paddle.nn.Linear(in_features=2048, out_features=2048)
        self.fc8 = paddle.nn.Linear(in_features=2048, out_features=2048)
        self.fc9 = paddle.nn.Linear(in_features=2048, out_features=1024)
        self.fc10 = paddle.nn.Linear(in_features=1024, out_features=512)
        self.fc11 = paddle.nn.Linear(in_features=512, out_features=256)
        self.fc12 = paddle.nn.Linear(in_features=256, out_features=128)
        self.fc13 = paddle.nn.Linear(in_features=128, out_features=64)
        self.fc14 = paddle.nn.Linear(in_features=64, out_features=1)

        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        x = self.relu(self.fc11(x))
        x = self.relu(self.fc12(x))
        x = self.relu(self.fc13(x))
        x = self.fc14(x)

        return x


# 声明定义好的线性回归模型
model = PUBGRegressor()

# 开启模型训练模式
model.train()

# 定义优化算法，使用随机梯度下降SGD
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.01, T_max=200)
# 不建议使用SGD，在这个数据量上收敛的很慢
# opt = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
# loss_func = paddle.nn.MSELoss()

train_df, test_df = read_data(train_datapath, test_datapath)

EPOCH_NUM = 200     # 大约100多个epoch收敛
BATCH_SIZE = 1000
training_data = train_df.iloc[:-10000].values.astype(np.float32)
val_data = train_df.iloc[-10000:].values.astype(np.float32)
min_loss = 100

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)

    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]

    train_loss = []
    for iter_id, mini_batch in enumerate(mini_batches):
        # 清空梯度变量，以备下一轮计算
        opt.clear_grad()

        x = np.array(mini_batch[:, :-1])
        y = np.array(mini_batch[:, -1:])

        # 将numpy数据转为飞桨动态图tensor的格式
        features = paddle.to_tensor(x)
        y = paddle.to_tensor(y)

        # 前向计算
        predicts = model(features)

        # 计算损失
        loss = paddle.nn.functional.l1_loss(predicts, label=y)
        # loss = loss_func(predicts, label=y)
        avg_loss = paddle.mean(loss)
        train_loss.append(avg_loss.numpy())

        # 反向传播，计算每层参数的梯度值
        avg_loss.backward()

        # 更新参数，根据设置好的学习率迭代一步
        opt.step()

    mini_batches = [val_data[k:k + BATCH_SIZE] for k in range(0, len(val_data), BATCH_SIZE)]
    val_loss = []
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])
        y = np.array(mini_batch[:, -1:])

        features = paddle.to_tensor(x)
        y = paddle.to_tensor(y)

        predicts = model(features)
        loss = paddle.nn.functional.l1_loss(predicts, label=y)
        # loss = loss_func(predicts, label=y)
        avg_loss = paddle.mean(loss)
        val_loss.append(avg_loss.numpy())

    print(f'Epoch {epoch_id}, train MAE {np.mean(train_loss)}, val MAE {np.mean(val_loss)}')

    if min_loss > np.mean(val_loss):
        min_loss = np.mean(val_loss)
        paddle.save(model.state_dict(), 'best-pubg.model')
        print("min loss: ", min_loss)

# 提交
model_file = '../models/pubg-loss4.539792.model'
model.set_state_dict(paddle.load(model_file))
model.eval()
test_data = paddle.to_tensor(test_df.values.astype(np.float32))
test_predict = model(test_data)
test_predict = test_predict.numpy().flatten()
test_predict = test_predict.round().astype(int)

pd.DataFrame({
    'team_placement': test_predict
}).to_csv('submission.csv', index=None)