import numpy as np
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
    feature_eng = np.concatenate([feature_data.values, feature_mean, feature_std], axis=1).astype(np.float32)

    feat_min = feature_eng.min(axis=0, keepdims=True)
    feat_max = feature_eng.max(axis=0, keepdims=True)
    feature_eng = (feature_eng - feat_min) / (feat_max - feat_min + 1e-8)

    feature = np.reshape(feature_eng, (opt.height, opt.width, feature_eng.shape[1]))

    # 加载流量数据（仅支持单流量）
    if opt.traffic == 'sms':
        data = f['data'][:, :, 0]
    elif opt.traffic == 'call':
        data = f['data'][:, :, 1]
    elif opt.traffic == 'internet':
        data = f['data'][:, :, 2]
    else:
        raise ValueError(f"未知流量类型: {opt.traffic}")

    data = np.asarray(data, dtype=np.float32)
    result = data.reshape((-1, 1, opt.height, opt.width))

    # 空间裁剪
    if opt.crop:
        result = result[:, :, opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1]]
        feature = feature[opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1], :]

    return result, feature.astype(np.float32)


def _build_supervised_samples(data_scaled, index, opt):

    close_size = opt.close_size
    horizon = getattr(opt, "horizon", 1)

    X, y, target_times = [], [], []

    for end in range(close_size, len(data_scaled) - horizon + 1):
        target_idx = end + horizon - 1

        xseq = data_scaled[end - close_size:end]

        X.append(xseq)
        y.append(data_scaled[target_idx])
        target_times.append(index[target_idx])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    X_meta = get_time_features(target_times)

    return X, X_meta, y


# 读取训练/验证/测试数据
def read_data(path, feature_path, opt):
    with h5py.File(path, 'r') as f:
        data, feature_data = traffic_loader(f, feature_path, opt)
        index = to_datetime(f['idx'][()].astype(str), format='%Y-%m-%d %H:%M')

    horizon = getattr(opt, "horizon", 1)

    # 测试集目标对应最后 opt.test_size 个时间点
    train_boundary = len(data) - opt.test_size

    if train_boundary < opt.close_size + horizon:
        raise ValueError(
            f"数据长度不足：len(data)={len(data)}, "
            f"train_boundary={train_boundary}, "
            f"close_size={opt.close_size}, horizon={horizon}"
        )

    mmn = MinMaxNorm01()
    mmn.fit(data[:train_boundary])
    data_scaled = mmn.transform(data)

    X, X_meta, y = _build_supervised_samples(data_scaled, index, opt)

    # 静态跨域特征: (C, H, W)
    feat = np.moveaxis(feature_data, -1, 0).astype(np.float32)
    X_crossdata = np.repeat(feat[np.newaxis, ...], X.shape[0], axis=0).astype(np.float32)

    return X, X_meta, X_crossdata, y, mmn


def read_test_data(path, feature_path, opt, mmn):
    with h5py.File(path, 'r') as f:
        data, feature_data = traffic_loader(f, feature_path, opt)
        index = to_datetime(f['idx'][()].astype(str), format='%Y-%m-%d %H:%M')

    data_scaled = mmn.transform(data)

    X_all, X_meta_all, y_all = _build_supervised_samples(data_scaled, index, opt)

    X_test = X_all[-opt.test_size:]
    X_meta_test = X_meta_all[-opt.test_size:]
    y_test = y_all[-opt.test_size:]

    feat = np.moveaxis(feature_data, -1, 0).astype(np.float32)
    X_cross_all = np.repeat(feat[np.newaxis, ...], X_all.shape[0], axis=0).astype(np.float32)
    X_cross = X_cross_all[-opt.test_size:]

    return X_test, X_meta_test, X_cross, y_test
