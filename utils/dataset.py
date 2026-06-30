import numpy as np
import torch
import pandas as pd
from pandas import to_datetime
import h5py
from datetime import datetime


# 最小最大归一化
class MinMaxNorm01(object):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()

    def transform(self, x):
        return (x - self.min) / (self.max - self.min + 1e-8)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        return x * (self.max - self.min) + self.min


# 时间特征提取（4维核心特征）
def get_time_features(index_list):
    holiday_start = datetime(2013, 11, 1)
    holiday_end = datetime(2014, 1, 1)

    features = []
    for idx in index_list:
        weekday = idx.weekday()
        is_weekend = 1.0 if weekday >= 5 else 0.0
        is_weekday = 1.0 if weekday < 5 else 0.0
        is_milan_holiday = 1.0 if (holiday_start <= idx <= holiday_end) else 0.0
        weekday_norm = weekday / 6.0
        features.append([weekday_norm, is_weekend, is_weekday, is_milan_holiday])
    return np.array(features, dtype=np.float32)


# 流量数据加载
def traffic_loader(f, feature_path, opt):
    feature_names = ['social', 'BSs', 'POI_Transportation', 'POI_Other']
    feature_data = pd.read_csv(feature_path, header=0)
    feature_data.columns = feature_names

    # 特征工程
    feature_mean = feature_data.mean(axis=1).values.reshape(-1, 1)
    feature_std = feature_data.std(axis=1).values.reshape(-1, 1)
    feature_eng = np.concatenate([feature_data.values, feature_mean, feature_std], axis=1)
    feature = np.reshape(feature_eng, (opt.height, opt.width, feature_eng.shape[1]))

    # 加载流量数据（仅支持单流量）
    if opt.traffic == 'sms':
        data = f['data'][:, :, 0]
    elif opt.traffic == 'call':
        data = f['data'][:, :, 1]
    elif opt.traffic == 'internet':
        data = f['data'][:, :, 2]
    else:
        raise ValueError("未知流量类型")

    result = data.reshape((-1, 1, opt.height, opt.width))

    # 空间裁剪
    if opt.crop:
        result = result[:, :, opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1]]
        feature = feature[opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1], :]

    return result, feature


# 读取训练/验证数据
def read_data(path, feature_path, opt):
    f = h5py.File(path, 'r')
    data, feature_data = traffic_loader(f, feature_path, opt)
    index = to_datetime(f['idx'][()].astype(str), format='%Y-%m-%d %H:%M')

    # 归一化
    mmn = MinMaxNorm01()
    data_scaled = mmn.fit_transform(data)

    # 时间元数据
    valid_index = index[opt.close_size:]
    X_meta = get_time_features(valid_index)

    # 构建时序窗口
    X, y = [], []
    for i in range(opt.close_size, len(data)):
        xseq = [data_scaled[i - c] for c in range(1, opt.close_size + 1)]
        X.append(xseq)
        y.append(data_scaled[i])

    X, y = np.asarray(X), np.asarray(y)

    # 静态跨域特征
    feat = np.moveaxis(feature_data, -1, 0)
    X_crossdata = np.repeat(feat[np.newaxis, ...], X.shape[0], axis=0)

    f.close()
    return X, X_meta, X_crossdata, y, mmn


# 读取测试数据
def read_test_data(path, feature_path, opt, mmn):
    f = h5py.File(path, 'r')
    data, feature_data = traffic_loader(f, feature_path, opt)
    index = to_datetime(f['idx'][()].astype(str), format='%Y-%m-%d %H:%M')
    data_scaled = mmn.transform(data)

    # 构建时序窗口
    X_test, y_test = [], []
    for i in range(opt.close_size, len(data)):
        xseq = [data_scaled[i - c] for c in range(1, opt.close_size + 1)]
        X_test.append(xseq)
        y_test.append(data_scaled[i])

    X_test, y_test = np.asarray(X_test), np.asarray(y_test)

    # 静态跨域特征
    feat = np.moveaxis(feature_data, -1, 0)
    X_cross = np.repeat(feat[np.newaxis, ...], X_test.shape[0], axis=0)

    # 时间元数据
    valid_index = index[opt.close_size:]
    X_meta_test = get_time_features(valid_index)

    f.close()
    return X_test, X_meta_test, X_cross, y_test
