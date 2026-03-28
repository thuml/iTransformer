import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_mf import Exp_Long_Term_Forecast_MF
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np
from utils.tools import parse_csv_list, parse_group_mapping, parse_float_mapping, parse_int_list


def _prepare_mf_args(args):
    args.mf_freqs_list = parse_csv_list(args.mf_freqs)
    args.mf_seq_lens_list = parse_int_list(args.mf_seq_lens)
    args.mf_pred_lens_list = parse_int_list(args.mf_pred_lens) if args.mf_pred_lens else []
    args.mf_var_groups_map = parse_group_mapping(args.mf_var_groups)
    args.mf_target_groups_map = parse_group_mapping(args.mf_target_groups)
    args.mf_loss_weights_map = parse_float_mapping(args.mf_loss_weights)

    if not args.mf_target_groups_map:
        args.mf_target_groups_map = dict(args.mf_var_groups_map)

    if not args.mf_freqs_list:
        raise ValueError('mf_freqs is empty.')
    if len(args.mf_freqs_list) != len(args.mf_seq_lens_list):
        raise ValueError('mf_seq_lens must match mf_freqs length.')
    if args.mf_pred_lens_list and len(args.mf_pred_lens_list) != len(args.mf_freqs_list):
        raise ValueError('mf_pred_lens must match mf_freqs length when provided.')
    for freq in args.mf_freqs_list:
        if freq not in args.mf_var_groups_map:
            raise ValueError('Missing variable group for frequency: {}'.format(freq))

    args.mf_seq_lens_map = {
        freq: seq_len for freq, seq_len in zip(args.mf_freqs_list, args.mf_seq_lens_list)
    }

    if args.mf_pred_lens_list:
        args.mf_pred_lens_map = {
            freq: pred_len for freq, pred_len in zip(args.mf_freqs_list, args.mf_pred_lens_list)
        }
    else:
        args.mf_pred_lens_map = {freq: args.pred_len for freq in args.mf_freqs_list}

    for freq in args.mf_freqs_list:
        if freq not in args.mf_loss_weights_map:
            args.mf_loss_weights_map[freq] = 1.0

    return args


def _validate_experiment_pairing(args):
    """
    Validate experiment name and model compatibility.
    Raises ValueError on invalid combination.
    """
    # Multi-frequency experiment requires the mixed-frequency model
    if args.exp_name == 'multi_train' and args.model != 'MfITransformer':
        raise ValueError("exp_name 'multi_train' requires --model MfITransformer.")

    # MfITransformer is only meaningful when running the multi-frequency experiment
    if args.model == 'MfITransformer' and args.exp_name != 'multi_train':
        raise ValueError("model 'MfITransformer' requires --exp_name multi_train.")

    return True

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, MfITransformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train, multi_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # Multi-frequency extension (mf mode is enabled when exp_name == 'multi_train')
    parser.add_argument('--mf_freqs', type=str, default='1h,1d',
                        help='comma-separated frequencies, e.g. 15min,1h,1d')
    parser.add_argument('--mf_seq_lens', type=str, default='96,7',
                        help='comma-separated input lengths aligned with mf_freqs')
    parser.add_argument('--mf_pred_lens', type=str, default='',
                        help='optional comma-separated pred lengths aligned with mf_freqs')
    parser.add_argument('--mf_var_groups', type=str,
                        default='1h:TEMP|PRES|DEWP|RAIN|WSPM;1d:PM2.5|PM10|SO2|NO2|CO|O3',
                        help='freq to variables map: freq:var1|var2;freq2:var3|var4')
    parser.add_argument('--mf_target_groups', type=str, default='',
                        help='optional freq to target variables map, defaults to mf_var_groups')
    parser.add_argument('--mf_loss_weights', type=str, default='',
                        help='optional freq to loss weight map, e.g. 1h:1.0;1d:1.0')
    parser.add_argument('--mf_anchor_freq', type=str, default='',
                        help='optional anchor frequency, defaults to the first in mf_freqs')

    args = parser.parse_args()
    
    if args.exp_name == 'multi_train':
        args = _prepare_mf_args(args)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # Validate that selected experiment and model are compatible
    _validate_experiment_pairing(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    elif args.exp_name == 'multi_train': # Multi-frequency extension experiment
        Exp = Exp_Long_Term_Forecast_MF
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
