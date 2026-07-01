import os
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

from dataset import read_data
from MDSTN import MDSTN


torch.manual_seed(22)
np.random.seed(22)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def moving_average_smooth(pred, window_size=3):
    if window_size <= 1:
        return pred

    pad = window_size // 2
    pred_padded = np.pad(
        pred,
        ((pad, pad), (0, 0), (0, 0), (0, 0)),
        mode='edge'
    )

    smoothed = pred.copy()
    for i in range(pad, len(pred_padded) - pad):
        smoothed[i - pad] = np.mean(
            pred_padded[i - pad:i + pad + 1],
            axis=0
        )

    return smoothed


def add_bool_arg(parser, name, default=True):
    """
    同时支持:
        -use_cross
        --use_cross
        --use-cross
        --no-use_cross
        --no-use-cross
    """
    group = parser.add_mutually_exclusive_group(required=False)

    pos_aliases = [f'-{name}', f'--{name}']
    dashed_pos = f'--{name.replace("_", "-")}'
    if dashed_pos not in pos_aliases:
        pos_aliases.append(dashed_pos)

    neg_aliases = [f'--no-{name}']
    dashed_neg = f'--no-{name.replace("_", "-")}'
    if dashed_neg not in neg_aliases:
        neg_aliases.append(dashed_neg)

    group.add_argument(*pos_aliases, dest=name, action='store_true')
    group.add_argument(*neg_aliases, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-height', type=int, default=100)
    parser.add_argument('-width', type=int, default=100)

    parser.add_argument('-traffic', type=str, default='sms',
                        choices=['sms', 'call', 'internet'])
    parser.add_argument('-nb_flow', type=int, default=1)
    parser.add_argument('-close_size', type=int, default=24)
    parser.add_argument('-horizon', type=int, default=1)

    parser.add_argument('-loss', type=str, default='l1', choices=['l1', 'mse'])
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-epoch_size', type=int, default=300)

    parser.add_argument('-rows', nargs='+', type=int, default=[40, 60])
    parser.add_argument('-cols', nargs='+', type=int, default=[40, 60])

    add_bool_arg(parser, 'crop', default=True)
    add_bool_arg(parser, 'train', default=True)

    parser.add_argument('-test_size', type=int, default=24 * 7)
    parser.add_argument('-save_dir', type=str, default='results')

    # 功能开关
    add_bool_arg(parser, 'use_meta', default=False)
    add_bool_arg(parser, 'use_cross', default=True)
    add_bool_arg(parser, 'use_causal_conv', default=True)
    add_bool_arg(parser, 'temporal_use_transformer', default=False)
    add_bool_arg(parser, 'use_dense_conv', default=True)
    add_bool_arg(parser, 'spatial_use_transformer', default=False)

    parser.add_argument('-fusion_mode', type=int, default=0)

    parser.add_argument('-smooth_window', type=int, default=1)

    return parser.parse_args()


def log(fname, s):
    with open(fname, 'a', encoding='utf-8') as f:
        f.write(f'{datetime.now()}: {s}\n')


def unpack_batch(batch, opt):

    if opt.use_cross and opt.use_meta:
        x, cross, meta, target = batch
    elif opt.use_cross:
        x, cross, target = batch
        meta = None
    elif opt.use_meta:
        x, meta, target = batch
        cross = None
    else:
        x, target = batch
        cross = None
        meta = None

    return x, cross, meta, target


def run_epoch(model, loader, criterion, opt, optimizer=None):
    total_loss = 0.0

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            x, cross, meta, target = unpack_batch(batch, opt)

            x = x.float().to(device)
            target = target.float().to(device)
            cross = cross.float().to(device) if cross is not None else None
            meta = meta.float().to(device) if meta is not None else None

            pred, _ = model(x, cross, meta)
            loss = criterion(pred, target)

            total_loss += loss.item()

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

    return total_loss / max(len(loader), 1)


def train_model(model, train_loader, valid_loader, criterion, opt, mmn):
    best_loss = float('inf')
    train_loss, valid_loss = [], []
    early_stop = 0
    patience = 30

    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(0.5 * opt.epoch_size),
            int(0.75 * opt.epoch_size)
        ],
        gamma=0.1
    )

    total_start = time.time()

    for epoch in range(opt.epoch_size):
        t_loss = run_epoch(model, train_loader, criterion, opt, optimizer)
        v_loss = run_epoch(model, valid_loader, criterion, opt)

        train_loss.append(t_loss)
        valid_loss.append(v_loss)
        scheduler.step()

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

        if epoch % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{opt.epoch_size} | '
                f'Train: {t_loss:.6f} | '
                f'Valid: {v_loss:.6f} | '
                f'Best: {best_loss:.6f}'
            )

        log(
            f'{opt.model_filename}.log',
            f'Epoch {epoch + 1} | Train: {t_loss:.6f} | Valid: {v_loss:.6f}'
        )

    total_time = time.time() - total_start
    print(f'总训练时间：{total_time // 3600:.0f}h {total_time % 3600 // 60:.0f}m')

    return train_loss, valid_loss


def predict(model, test_loader, mmn, opt):
    model.eval()
    preds, truths, gates = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x, cross, meta, target = unpack_batch(batch, opt)

            x = x.float().to(device)
            cross = cross.float().to(device) if cross is not None else None
            meta = meta.float().to(device) if meta is not None else None

            pred, gate = model(x, cross, meta)

            preds.append(pred.cpu().numpy())
            truths.append(target.cpu().numpy())

            if gate is not None:
                b = x.shape[0]
                h, w = target.shape[-2], target.shape[-1]

                gate_np = (
                    gate
                    .transpose(1, 2)
                    .contiguous()
                    .reshape(b, -1, h, w)
                    .cpu()
                    .numpy()
                )
                gates.append(gate_np)

    pred_norm = np.concatenate(preds, axis=0)
    truth_norm = np.concatenate(truths, axis=0)
    gate_np = np.concatenate(gates, axis=0) if gates else None

    if opt.smooth_window > 1:
        pred_norm = moving_average_smooth(pred_norm, opt.smooth_window)

    pred = mmn.inverse_transform(pred_norm).clip(min=0)
    truth = mmn.inverse_transform(truth_norm)

    return pred, truth, gate_np


def train_valid_split(dataset, test_size=0.1):
    idx = np.random.permutation(len(dataset))
    split = int(len(idx) * test_size)
    return idx[split:], idx[:split]


def build_dataset(x, cross, meta, y, opt):
    if opt.use_cross and opt.use_meta:
        return list(zip(x, cross, meta, y))
    elif opt.use_cross:
        return list(zip(x, cross, y))
    elif opt.use_meta:
        return list(zip(x, meta, y))
    else:
        return list(zip(x, y))


if __name__ == '__main__':
    opt = get_args()

 
    DATA_PATH = "/root/autodl-fs/data_git_version.h5"
    FEATURE_PATH = "/root/autodl-fs/11.MVSTGN-main/MVSTGN-main/data/crawled_feature.csv"


    exp_flag = (
        f"causal_{opt.use_causal_conv}_"
        f"temp_{'trans' if opt.temporal_use_transformer else 'mamba'}_"
        f"dense_{opt.use_dense_conv}_"
        f"spat_{'trans' if opt.spatial_use_transformer else 'mamba'}_"
        f"fusion_{opt.fusion_mode}"
    )

    opt.save_dir = f"{opt.save_dir}/{opt.traffic}/MDSTN/{exp_flag}"
    os.makedirs(opt.save_dir, exist_ok=True)
    opt.model_filename = f"{opt.save_dir}/MDSTN_{opt.traffic}"

    # 加载数据
    X, X_meta, X_cross, y, mmn = read_data(DATA_PATH, FEATURE_PATH, opt)

    if not opt.use_meta:
        X_meta = None
    if not opt.use_cross:
        X_cross = None

    # 数据切分：最后 opt.test_size 个样本作为测试集
    x_train, x_test = X[:-opt.test_size], X[-opt.test_size:]
    y_train, y_test = y[:-opt.test_size], y[-opt.test_size:]

    if opt.use_meta:
        meta_train, meta_test = X_meta[:-opt.test_size], X_meta[-opt.test_size:]
    else:
        meta_train, meta_test = None, None

    if opt.use_cross:
        cross_train, cross_test = X_cross[:-opt.test_size], X_cross[-opt.test_size:]
    else:
        cross_train, cross_test = None, None

    train_dataset = build_dataset(
        x_train,
        cross_train,
        meta_train,
        y_train,
        opt
    )

    test_dataset = build_dataset(
        x_test,
        cross_test,
        meta_test,
        y_test,
        opt
    )

    train_idx, valid_idx = train_valid_split(train_dataset, test_size=0.1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=0,
        drop_last=True
    )

    valid_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=SubsetRandomSampler(valid_idx),
        num_workers=0,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    model = MDSTN(
        input_shape=X.shape,
        meta_shape=X_meta.shape if opt.use_meta else None,
        cross_shape=X_cross.shape if opt.use_cross else None,
        nb_flows=opt.nb_flow,
        use_causal_conv=opt.use_causal_conv,
        temporal_use_transformer=opt.temporal_use_transformer,
        use_dense_conv=opt.use_dense_conv,
        spatial_use_transformer=opt.spatial_use_transformer,
        fusion_mode=opt.fusion_mode
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"总参数量：{total_params:,} | 可训练：{trainable_params:,}")

    criterion = nn.L1Loss() if opt.loss == 'l1' else nn.MSELoss()

    if opt.train:
        print("开始训练...")
        train_loss, valid_loss = train_model(
            model,
            train_loader,
            valid_loader,
            criterion,
            opt,
            mmn
        )

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='训练损失')
        plt.plot(valid_loss, label='验证损失')
        plt.legend()
        plt.savefig(f"{opt.save_dir}/loss_curve.png")
        plt.close()

        np.savez(
            f"{opt.save_dir}/loss_data.npz",
            train=np.asarray(train_loss),
            valid=np.asarray(valid_loss)
        )

    print("开始测试...")
    model.load_state_dict(
        torch.load(f'{opt.model_filename}.pth', map_location=device)
    )

    pred, truth, gate_z = predict(model, test_loader, mmn, opt)

    rmse = np.sqrt(metrics.mean_squared_error(truth.ravel(), pred.ravel()))
    mae = metrics.mean_absolute_error(truth.ravel(), pred.ravel())
    r2 = metrics.r2_score(truth.ravel(), pred.ravel())

    print(f'测试结果：RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}')

    save_dict = {
        'pred': pred,
        'truth': truth,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'params': total_params
    }

    if gate_z is not None:
        save_dict['gate_z'] = gate_z

    np.savez(f"{opt.save_dir}/predictions.npz", **save_dict)

    with open(f"{opt.save_dir}/metrics.txt", 'w', encoding='utf-8') as f:
        f.write(
            f"RMSE: {rmse:.4f}\n"
            f"MAE: {mae:.4f}\n"
            f"R2: {r2:.4f}\n"
            f"总参数量: {total_params:,}\n"
            f"可训练参数量: {trainable_params:,}\n"
        )
