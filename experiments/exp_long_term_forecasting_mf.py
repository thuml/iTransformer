from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from typing import Dict

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_MF(Exp_Basic):
    """Mixed-frequency long-term forecasting experiment.

    This class is intentionally MF-only and mirrors the SF training flow:
    - Batches are dicts keyed by frequency (e.g., `1h`, `1d`).
    - Decoder input is built per frequency using `label_len` history + zero future.
    - Model forward uses encoder-decoder signature and returns dict outputs.
    - Shared encoder/decoder are used in the model; outputs/metrics stay per frequency.

    The regular single-frequency behavior remains in
    `experiments/exp_long_term_forecasting.py`.
    """

    def __init__(self, args):
        """Initialize multi-frequency forecasting experiment with parsed CLI args."""
        super(Exp_Long_Term_Forecast_MF, self).__init__(args)

    def _build_model(self):
        """Instantiate model by name and optionally wrap with DataParallel."""
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """Create dataset/dataloader pair for split: train, val, test, or pred."""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """Create optimizer used for training."""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Return the base regression loss."""
        criterion = nn.MSELoss()
        return criterion

    def _to_device(self, obj):
        """Recursively move tensors (including nested dict values) to target device."""
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        if obj is None:
            return None
        return obj.float().to(self.device)

    def _prepare_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """Validate MF batch format and move all tensors to the selected device."""
        if not isinstance(batch_x, dict):
            raise ValueError('MF experiment expects dict batches keyed by frequency.')

        batch_x = self._to_device(batch_x)
        batch_y = self._to_device(batch_y)
        batch_x_mark = self._to_device(batch_x_mark)
        batch_y_mark = self._to_device(batch_y_mark)
        return batch_x, batch_y, batch_x_mark, batch_y_mark

    def _build_dec_input(self, batch_y: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build per-frequency decoder input as [history(label_len), zeros(pred_len)]."""
        dec_inp = {}
        label_len = self.args.label_len
        pred_lens = getattr(self.args, 'mf_pred_lens_map', {})

        for f_key, y_f in batch_y.items():
            pred_len_f = pred_lens.get(f_key, self.args.pred_len)
            if y_f.shape[1] < label_len + pred_len_f:
                raise ValueError(
                    'batch_y[{}] length {} is smaller than label_len + pred_len ({})'.format(
                        f_key, y_f.shape[1], label_len + pred_len_f
                    )
                )
            dec_zeros = torch.zeros_like(y_f[:, -pred_len_f:, :]).float()
            dec_inp[f_key] = torch.cat([y_f[:, :label_len, :], dec_zeros], dim=1).float().to(self.device)

        return dec_inp

    def _slice_pred_targets(self, batch_y: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract per-frequency prediction horizon targets from decoder-format targets."""
        targets = {}
        pred_lens = getattr(self.args, 'mf_pred_lens_map', {})
        for f_key, y_f in batch_y.items():
            pred_len_f = pred_lens.get(f_key, self.args.pred_len)
            targets[f_key] = y_f[:, -pred_len_f:, :]
        return targets

    def _forward_pass(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """Run MF encoder-decoder forward pass and unwrap attention output if requested."""
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.output_attention:
                    outputs = outputs[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
        return outputs

    def _compute_mf_loss(self, outputs, targets, criterion):
        """Aggregate weighted loss over all configured MF frequency heads."""
        loss = 0.0
        loss_weights = getattr(self.args, 'mf_loss_weights_map', {})
        for f_key, pred in outputs.items():
            if f_key not in targets:
                continue
            weight = loss_weights.get(f_key, 1.0)
            loss = loss + weight * criterion(pred, targets[f_key])
        return loss

    def vali(self, vali_data, vali_loader, criterion):
        """Evaluate average MF validation loss across the validation dataloader."""
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                dec_inp = self._build_dec_input(batch_y)
                targets = self._slice_pred_targets(batch_y)
                outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                loss = self._compute_mf_loss(outputs, targets, criterion)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """Train MF model using weighted per-frequency losses and early stopping."""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                dec_inp = self._build_dec_input(batch_y)
                targets = self._slice_pred_targets(batch_y)
                outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                loss = self._compute_mf_loss(outputs, targets, criterion)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        """Evaluate MF test performance and persist per-frequency predictions/metrics."""
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        mf_preds = {}
        mf_trues = {}

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                dec_inp = self._build_dec_input(batch_y)
                targets = self._slice_pred_targets(batch_y)
                outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                for f_key, pred in outputs.items():
                    if f_key not in mf_preds:
                        mf_preds[f_key] = []
                        mf_trues[f_key] = []
                    mf_preds[f_key].append(pred.detach().cpu().numpy())
                    mf_trues[f_key].append(targets[f_key].detach().cpu().numpy())

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open('result_long_term_forecast.txt', 'a')
        f.write(setting + ' (MF)\n')
        for f_key in mf_preds:
            pred_np = np.array(mf_preds[f_key])
            true_np = np.array(mf_trues[f_key])
            pred_np = pred_np.reshape(-1, pred_np.shape[-2], pred_np.shape[-1])
            true_np = true_np.reshape(-1, true_np.shape[-2], true_np.shape[-1])

            mae, mse, rmse, mape, mspe = metric(pred_np, true_np)
            print('[{}] mse:{}, mae:{}'.format(f_key, mse, mae))
            f.write('[{}] mse:{}, mae:{}\n'.format(f_key, mse, mae))

            np.save(folder_path + 'metrics_{}.npy'.format(f_key), np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred_{}.npy'.format(f_key), pred_np)
            np.save(folder_path + 'true_{}.npy'.format(f_key), true_np)
        f.write('\n')
        f.close()

        return

    def predict(self, setting, load=False):
        """Run MF inference and save one prediction file per frequency key."""
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                dec_inp = self._build_dec_input(batch_y)
                outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                folder_path = './results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                for f_key, pred in outputs.items():
                    pred_np = pred.detach().cpu().numpy()
                    np.save(folder_path + 'real_prediction_{}.npy'.format(f_key), pred_np)
                return

        return
