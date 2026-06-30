import numpy as np
import pandas as pd
import h5py


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


# 流量数据加载
def traffic_loader(f, feature_path, opt):
    feature_names = ['social', 'BSs', 'POI_Transportation', 'POI_Other']
    feature_data = pd.read_csv(feature_path, header=0)
    feature_data.columns = feature_names

    # 特征工程：原始静态特征 + 行均值 + 行标准差
    feature_mean = feature_data.mean(axis=1).values.reshape(-1, 1)
    feature_std = feature_data.std(axis=1).values.reshape(-1, 1)
    feature_eng = np.concatenate([feature_data.values, feature_mean, feature_std], axis=1).astype(np.float32)

    feature = np.reshape(feature_eng, (opt.height, opt.width, feature_eng.shape[1]))

    # 加载流量数据：仅支持单流量
    if opt.traffic == 'sms':
        data = f['data'][:, :, 0]
    elif opt.traffic == 'call':
        data = f['data'][:, :, 1]
    elif opt.traffic == 'internet':
        data = f['data'][:, :, 2]
    else:
        raise ValueError("未知流量类型，请使用 sms、call 或 internet")

    result = data.reshape((-1, 1, opt.height, opt.width)).astype(np.float32)

    # 空间裁剪
    if opt.crop:
        result = result[:, :, opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1]]
        feature = feature[opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1], :]

    return result, feature.astype(np.float32)


# 读取训练/验证数据

def read_data(path, feature_path, opt):
    with h5py.File(path, 'r') as f:
        data, feature_data = traffic_loader(f, feature_path, opt)

    # 归一化
    mmn = MinMaxNorm01()
    data_scaled = mmn.fit_transform(data).astype(np.float32)

    # 构建历史窗口

    X, y = [], []
    for i in range(opt.close_size, len(data_scaled)):
        xseq = [data_scaled[i - c] for c in range(opt.close_size, 0, -1)]
        X.append(xseq)
        y.append(data_scaled[i])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # 静态跨域特征：[H, W, C] -> [C, H, W] -> [N, C, H, W]
    feat = np.moveaxis(feature_data, -1, 0).astype(np.float32)
    X_crossdata = np.repeat(feat[np.newaxis, ...], X.shape[0], axis=0).astype(np.float32)

    return X, X_crossdata, y, mmn


# 读取测试数据

def read_test_data(path, feature_path, opt, mmn):
    with h5py.File(path, 'r') as f:
        data, feature_data = traffic_loader(f, feature_path, opt)

    data_scaled = mmn.transform(data).astype(np.float32)

    X_test, y_test = [], []
    for i in range(opt.close_size, len(data_scaled)):
        xseq = [data_scaled[i - c] for c in range(opt.close_size, 0, -1)]
        X_test.append(xseq)
        y_test.append(data_scaled[i])

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    feat = np.moveaxis(feature_data, -1, 0).astype(np.float32)
    X_cross = np.repeat(feat[np.newaxis, ...], X_test.shape[0], axis=0).astype(np.float32)

    return X_test, X_cross, y_test
