from utils.tools import dotdict
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np



fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


arg = dotdict()

# NEW OPTIONS : #

arg.test_size = None                 # default is 0.2 which makes the training 0.7 ! #
arg.kind_of_scaler = None            # default is 'Standard'. Another Option is 'MinMax' (recommended) #
arg.name_of_col_with_date = None     # default is 'date'. Name of your date column in your dataset #

arg.kind_of_optim = 'default'        # default is 'Adam'.
                                     # other options : 'AdamW', 'SparseAdam', 'SGD', 'RMSprop', 'RAdam', 'NAdam' ,'LBFGS',
                                     #                     'Adamax' 'ASGD' 'Adadelta' 'Adagrad'

arg.criter = 'default'               # default is nn.MSELoss ( Mean Squared Error )
                                     # other options : 'wmape', 'smape', 'mae', 'rmse', 'quantileloss', 'huberloss', 'pinballloss'

# NEW Accessories : #

exp.trues_during_training
exp.preds_during_training
exp.train_losses
exp.test_losses

#####################

arg.is_training = 1                         # help: status
arg.model_id = 'test'

arg.model = 'iTransformer'                  # help: model name. options: iTransformer, iInformer, iReformer, iFlowformer, iFlashformer
arg.data = 'custom'                         # help: dataset type

arg.root_path = 'input/train'                         # help: main directory path of the data file
arg.data_path =  'data.csv'                      # help: name of data csv file

arg.target_root_path =  'input/test'
arg.target_data_path = 'data.csv'


arg.features = 'MS'                             # help: forecasting task , options: M ->multivariate predict multivariate , or
#                                                                                   S ->univariate predict univariate , or
#                                                                                   MS ->multivariate predict univariate

arg.target =  'Close'                           # help: target feature in S or MS task

arg.freq = 'b'                                      # help: Freq for time features encoding. options: s ->secondly , t ->minutely, h:hourly
#                                                                                                       d ->daily , w ->weekly, m ->monthly
#                                                                                                           b ->business days
#                                                                                                               also more detailed freq like 15min or 3h

arg.checkpoints =  './checkpoints/'                 # help: location to save model checkpoints

arg.seq_len = 1*5*3                                     # help: input sequence length
arg.label_len = 1*1                                         # help: start token length
arg.pred_len =  1*3                                         # help: prediction sequence length

arg.enc_in = 6                                         # help: encoder input size
arg.dec_in = 6                                         # help: decoder input size
arg.c_out = 1                                         # help: output size -> applicable on arbitrary number of variates in inverted Transformers
arg.d_model =  512                                        # help: dimension of model
arg.n_heads =  8                                        # help: num of heads
arg.e_layers = 8                                         # help: num of encoder layers
arg.d_layers = 8                                         # help: num of decoder layers
arg.d_ff =  2048                                        # help: dimension of fcn
arg.moving_avg = 25                                         # help: window size of moving average
arg.factor = 1                                         # help: attn factor
arg.distil = True                                         # help: whether to use distilling in encoder, using this argument means not using distilling

arg.dropout =  0.01

arg.embed = 'timeF'                                                 # help: time features encoding, options: timeF OR fixed OR learned
arg.activation =  'ReLU'                                               # help: Name of activation Function

#arg.output_attention =  None                                           # help: Whether to output attention in ecoder
#arg.do_predict = None                                                   # help: whether to predict unseen future data

arg.num_workers = 10                                                # help: data loader num workers
arg.itr = 1                                                         # help: How many times repeat experiments 

arg.train_epochs = 21

arg.batch_size = 16

arg.patience = 7                                                   # help: early stopping patience

arg.learning_rate = 0.00005

arg.des = 'test'                                                    # help: exp description

arg.loss = 'MSE'                                                    # help: loss function

arg.lradj = 'type1'                                                       # help: adjust learning rate
arg.use_amp =  False                                                     # help: use automatic mixed precision training

arg.use_gpu =  True if torch.cuda.is_available() else False             # help: whether to use gpu
arg.gpu =  0 # help: GPU
arg.use_multi_gpu = False
arg.devices = '0,1,2,3'

arg.exp_name = 'MTSF'

arg.channel_independence =  False                                        # help: whether to use channel_independence mechanism

arg.inverse =  True                                                         # help: inverse output data

arg.class_strategy =  'projection'                                        # help: options: projection/average/cls_token





arg.efficient_training = False                                      # help: whether to use efficient_training (exp_name should be partial_train) | See Figure 8

arg.use_norm = True                                                   # help: use norm and denorm | type=int

arg.partial_start_index =  0                        # help: the start index of variates for partial training,
#                                                       you can select [partial_start_index, min(enc_in + partial_start_index, N)]

#if arg.use_gpu and arg.use_multi_gpu:
#        arg.devices = arg.devices.replace(' ', '')
#        device_ids = arg.devices.split(',')
#        arg.device_ids = [int(id_) for id_ in device_ids]
#        arg.gpu = arg.device_ids[0]


print('Args in experiment:')
print(arg)



if input("Press Enter To Start :" ) == '' :
    pass
else:
    exit()

if arg.exp_name == 'partial_train':                                         # See Figure 8 of our paper, for the detail
    Exp = Exp_Long_Term_Forecast_Partial
else:                                                                       # MTSF: multivariate time series forecasting
    Exp = Exp_Long_Term_Forecast

if arg.is_training:
    for ii in range(arg.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                arg.model_id,
                arg.model,
                arg.data,
                arg.features,
                arg.seq_len,
                arg.label_len,
                arg.pred_len,
                arg.d_model,
                arg.n_heads,
                arg.e_layers,
                arg.d_layers,
                arg.d_ff,
                arg.factor,
                arg.embed,
                arg.distil,
                arg.des,
                arg.class_strategy, ii)
        
        exp = Exp(arg)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        
        if arg.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        
        torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            arg.model_id,
            arg.model,
            arg.data,
            arg.features,
            arg.seq_len,
            arg.label_len,
            arg.pred_len,
            arg.d_model,
            arg.n_heads,
            arg.e_layers,
            arg.d_layers,
            arg.d_ff,
            arg.factor,
            arg.embed,
            arg.distil,
            arg.des,
            arg.class_strategy, ii)
        
        exp = Exp(arg)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
#end#
