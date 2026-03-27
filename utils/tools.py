import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# Parsing utilities used by run.py for multifrequency CLI args
def parse_csv_list(raw_value):
    """Parse a comma-separated string into a list of stripped items.

    Uses pandas string split for robustness but returns a plain Python list.
    """
    if raw_value is None:
        return []
    s = str(raw_value)
    if not s.strip():
        return []
    # pandas split is tolerant of surrounding whitespace
    items = pd.Series([s]).str.split(',').iloc[0]
    return [item.strip() for item in items if str(item).strip()]


def parse_group_mapping(raw_value):
    """Parse mapping strings like '1h:A|B;1d:C|D' -> {'1h': ['A','B'], '1d': ['C','D']}"""
    mapping = {}
    if raw_value is None:
        return mapping
    s = str(raw_value).strip()
    if not s:
        return mapping
    entries = [entry.strip() for entry in s.split(';') if entry.strip()]
    for entry in entries:
        if ':' not in entry:
            continue
        key, value = entry.split(':', 1)
        items = [item.strip() for item in value.split('|') if item.strip()]
        mapping[key.strip()] = items
    return mapping


def parse_float_mapping(raw_value):
    """Parse mapping strings like '1h:1.0;1d:0.5' -> {'1h':1.0, '1d':0.5}"""
    mapping = {}
    if raw_value is None:
        return mapping
    s = str(raw_value).strip()
    if not s:
        return mapping
    entries = [entry.strip() for entry in s.split(';') if entry.strip()]
    for entry in entries:
        if ':' not in entry:
            continue
        key, value = entry.split(':', 1)
        try:
            mapping[key.strip()] = float(value.strip())
        except ValueError:
            # ignore invalid floats
            continue
    return mapping


def parse_int_list(raw_value):
    items = parse_csv_list(raw_value)
    out = []
    for it in items:
        try:
            out.append(int(it))
        except ValueError:
            try:
                out.append(int(float(it)))
            except ValueError:
                raise ValueError(f"Invalid integer value in list: {it}")
    return out
