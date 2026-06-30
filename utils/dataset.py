import numpy as np
import pandas as pd
import h5py
from datetime import datetime


class MinMaxNorm01:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()

    def transform(self, x):
        return (x - self.min) / (self.max - self.min + 1e-8)

    def inverse(self, x):
        return x * (self.max - self.min) + self.min


def traffic_loader(f, feature_path, opt):
    df = pd.read_csv(feature_path)

    feature = df.values.astype(np.float32)
    feature = feature.reshape(opt.height, opt.width, -1)

    data = f['data'][:, :, 0]  # single flow

    data = data.reshape(-1, opt.height, opt.width)

    return data, feature


def read_data(path, feature_path, opt):
    f = h5py.File(path, 'r')

    data, cross = traffic_loader(f, feature_path, opt)

    mmn = MinMaxNorm01()
    data = mmn.fit(data)

    X, y = [], []

    for i in range(opt.close_size, len(data)):
        X.append(data[i-opt.close_size:i])
        y.append(data[i])

    X = np.array(X, dtype=np.float32)[:, :, None, :, :]
    y = np.array(y, dtype=np.float32)[:, None, :, :]

    cross = np.repeat(
        cross[None, ...],
        len(X),
        axis=0
    ).transpose(0,3,1,2)

    return X, cross, y, mmn
