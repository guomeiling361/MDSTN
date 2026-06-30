import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import time
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from datetime import datetime

# 导入自定义模块
from dataset import read_data, read_test_data
from MDSTN import MCDAG

# 基础配置
torch.manual_seed(22)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]  # 支持中文绘图


# 模型参数量统计
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# 预测平滑
def moving_average_smooth(pred, window_size=3):
    pad = window_size // 2
    pred_padded = np.pad(pred, ((pad, pad), (0, 0), (0, 0), (0, 0)), mode='edge')
    for i in range(pad, len(pred_padded) - pad):
        pred[i - pad] = np.mean(pred_padded[i - pad:i + pad + 1], axis=0)
    return pred


# 命令行参数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-height', type=int, default=100)
    parser.add_argument('-width', type=int, default=100)
    parser.add_argument('-traffic', type=str, default='sms')
    parser.add_argument('-nb_flow', type=int, default=1)
    parser.add_argument('-close_size', type=int, default=24)
    parser.add_argument('-loss', type=str, default='l1')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-epoch_size', type=int, default=300)
    parser.add_argument('-rows', nargs='+', type=int, default=[40, 60])
    parser.add_argument('-cols', nargs='+', type=int, default=[40, 60])
    parser.add_argument('-crop', action='store_true', default=True)
    parser.add_argument('-train', action='store_true', default=True)
    parser.add_argument('-test_size', type=int, default=24 * 7)
    parser.add_argument('-save_dir', type=str, default='results')

    # 功能开关
    parser.add_argument('-use_meta', action='store_true', default=False)
    parser.add_argument('-use_cross', action='store_true', default=True)
    parser.add_argument('-use_causal_conv', action='store_true', default=True)
    parser.add_argument('-temporal_use_transformer', action='store_true', default=False)
    parser.add_argument('-use_dense_conv', action='store_true', default=True)
    parser.add_argument('-spatial_use_transformer', action='store_true', default=False)
    parser.add_argument('-fusion_mode', type=int, default=0)
    return parser.parse_args()


# 日志记录
def log(fname, s):
    with open(fname, 'a', encoding='utf-8') as f:
        f.write(f'{datetime.now()}: {s}\n')


# 单轮训练/验证
def run_epoch(model, loader, criterion, optimizer=None):
    total_loss = 0.0
    model.train() if optimizer else model.eval()

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            if len(batch) == 4:
                x, cross, meta, target = batch
            elif len(batch) == 3:
                x, meta, target = batch
                cross = None
            else:
                x, target = batch
                meta, cross = None, None

            x = x.float().to(device)
            target = target.float().to(device)
            meta = meta.float().to(device) if meta is not None else None
            cross = cross.float().to(device) if cross is not None else None

            pred, _ = model(x, cross, meta)
            loss = criterion(pred, target)
            total_loss += loss.item()

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

    return total_loss / len(loader)


# 训练主函数
def train_model(model, train_loader, valid_loader, criterion, opt):
    best_loss = float('inf')
    train_loss, valid_loss = [], []
    early_stop = 0
    patience = 30

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(0.5 * opt.epoch_size), int(0.75 * opt.epoch_size)],
                                               gamma=0.1)

    total_start = time.time()
    for epoch in range(opt.epoch_size):
        t_loss = run_epoch(model, train_loader, criterion, optimizer)
        v_loss = run_epoch(model, valid_loader, criterion)

        train_loss.append(t_loss)
        valid_loss.append(v_loss)
        scheduler.step()

        # 保存最优模型
        if v_loss < best_loss:
            best_loss = v_loss
            early_stop = 0
            torch.save(model.state_dict(), f'{opt.model_filename}.pth')
            with open(f'{opt.model_filename}_mmn.pkl', 'wb') as f:
                pickle.dump(mmn, f)
        else:
            early_stop += 1
            if early_stop >= patience:
                print(f"早停：{patience}轮无提升")
                break

        # 日志
        if epoch % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{opt.epoch_size} | Train: {t_loss:.6f} | Valid: {v_loss:.6f} | Best: {best_loss:.6f}')
        log(f'{opt.model_filename}.log', f'Epoch {epoch + 1} | Train: {t_loss:.6f} | Valid: {v_loss:.6f}')

    # 训练时间统计
    total_time = time.time() - total_start
    print(f'总训练时间：{total_time // 3600:.0f}h {total_time % 3600 // 60:.0f}m')
    return train_loss, valid_loss


# 预测
def predict(model, test_loader, mmn, opt):
    model.eval()
    preds, truths, gates = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                x, cross, meta, target = batch
            elif len(batch) == 3:
                x, meta, target = batch
                cross = None
            else:
                x, target = batch
                meta, cross = None, None

            x = x.float().to(device)
            meta = meta.float().to(device) if meta is not None else None
            cross = cross.float().to(device) if cross is not None else None

            pred, gate = model(x, cross, meta)
            preds.append(pred.cpu().numpy())
            truths.append(target.cpu().numpy())
            if gate is not None:
                gates.append(gate.transpose(1, 2).view(x.shape[0], -1, opt.rows[1] - opt.rows[0],
                                                       opt.cols[1] - opt.cols[0]).cpu().numpy())

    # 拼接数据
    pred_norm = moving_average_smooth(np.concatenate(preds))
    truth_norm = np.concatenate(truths)
    gate_np = np.concatenate(gates) if gates else None

    # 反归一化 + 合理后处理（流量非负）
    pred = mmn.inverse_transform(pred_norm).clip(min=0)
    truth = mmn.inverse_transform(truth_norm)
    return pred, truth, gate_np


# 数据集划分
def train_valid_split(dataset, test_size=0.1):
    idx = np.random.permutation(len(dataset))
    split = int(len(idx) * test_size)
    return idx[split:], idx[:split]


if __name__ == '__main__':
    opt = get_args()
    # ===================== 请修改这里的路径 =====================
    DATA_PATH = "/root/autodl-fs/data_git_version.h5"
    FEATURE_PATH = "/root/autodl-fs/11.MVSTGN-main/MVSTGN-main/data/crawled_feature.csv"
    # ===========================================================

    # 实验路径配置
    exp_flag = f"causal_{opt.use_causal_conv}_temp_{'trans' if opt.temporal_use_transformer else 'mamba'}_dense_{opt.use_dense_conv}_spat_{'trans' if opt.spatial_use_transformer else 'mamba'}_fusion_{opt.fusion_mode}"
    opt.save_dir = f"{opt.save_dir}/{opt.traffic}/MCDAG/{exp_flag}"
    os.makedirs(opt.save_dir, exist_ok=True)
    opt.model_filename = f"{opt.save_dir}/MCDAG_{opt.traffic}"

    # 加载数据
    X, X_meta, X_cross, y, mmn = read_data(DATA_PATH, FEATURE_PATH, opt)
    if not opt.use_meta: X_meta = None
    if not opt.use_cross: X_cross = None

    # 数据集分割
    x_train, x_test = X[:-opt.test_size], X[-opt.test_size:]
    y_train, y_test = y[:-opt.test_size], y[-opt.test_size:]
    meta_train, meta_test = (X_meta[:-opt.test_size], X_meta[-opt.test_size:]) if opt.use_meta else (None, None)
    cross_train, cross_test = (X_cross[:-opt.test_size], X_cross[-opt.test_size:]) if opt.use_cross else (None, None)

    if opt.use_meta and opt.use_cross:
        # 正确顺序：x, cross, meta, y
        train_dataset = list(zip(x_train, cross_train, meta_train, y_train))
        test_dataset = list(zip(x_test, cross_test, meta_test, y_test))

    elif opt.use_meta:
        # 没 cross 但有 meta
        train_dataset = list(zip(x_train, meta_train, y_train))
        test_dataset = list(zip(x_test, meta_test, y_test))

    elif opt.use_cross:
         # 只有 cross
        train_dataset = list(zip(x_train, cross_train, y_train))
        test_dataset = list(zip(x_test, cross_test, y_test))

    else:
        train_dataset = list(zip(x_train, y_train))
        test_dataset = list(zip(x_test, y_test))

        # =====================================================================

    # 数据加载器
    train_idx, valid_idx = train_valid_split(train_dataset)
    train_loader = DataLoader(train_dataset, opt.batch_size, sampler=SubsetRandomSampler(train_idx), num_workers=0,
                              drop_last=True)
    valid_loader = DataLoader(train_dataset, opt.batch_size, sampler=SubsetRandomSampler(valid_idx), num_workers=0)
    test_loader = DataLoader(test_dataset, opt.batch_size, shuffle=False, num_workers=0)

    # 初始化模型
    model = MCDAG(
        input_shape=X.shape, meta_shape=X_meta.shape if opt.use_meta else None,
        cross_shape=X_cross.shape if opt.use_cross else None, nb_flows=opt.nb_flow,
        use_causal_conv=opt.use_causal_conv, temporal_use_transformer=opt.temporal_use_transformer,
        use_dense_conv=opt.use_dense_conv, spatial_use_transformer=opt.spatial_use_transformer,
        fusion_mode=opt.fusion_mode
    ).to(device)

    # 打印参数量
    total_params, trainable_params = count_parameters(model)
    print(f"总参数量：{total_params:,} | 可训练：{trainable_params:,}")

    # 损失函数
    criterion = nn.L1Loss() if opt.loss == 'l1' else nn.MSELoss()

    # 训练
    if opt.train:
        print("开始训练...")
        train_loss, valid_loss = train_model(model, train_loader, valid_loader, criterion, opt)

        # 保存损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='训练损失')
        plt.plot(valid_loss, label='验证损失')
        plt.legend()
        plt.savefig(f"{opt.save_dir}/loss_curve.png")
        np.savez(f"{opt.save_dir}/loss_data.npz", train=train_loss, valid=valid_loss)

    # 测试
    print("开始测试...")
    model.load_state_dict(torch.load(f'{opt.model_filename}.pth', map_location=device))
    pred, truth, gate_z = predict(model, test_loader, mmn, opt)

    # 计算指标
    rmse = np.sqrt(metrics.mean_squared_error(truth.ravel(), pred.ravel()))
    mae = metrics.mean_absolute_error(truth.ravel(), pred.ravel())
    r2 = metrics.r2_score(truth.ravel(), pred.ravel())

    print(f'测试结果：RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}')

    # 保存结果
    save_dict = {'pred': pred, 'truth': truth, 'rmse': rmse, 'mae': mae, 'r2': r2, 'params': total_params}
    if gate_z is not None: save_dict['gate_z'] = gate_z
    np.savez(f"{opt.save_dir}/predictions.npz", **save_dict)

    with open(f"{opt.save_dir}/metrics.txt", 'w', encoding='utf-8') as f:
        f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\n总参数量: {total_params:,}")
