import os
import time
import torch
import tempfile
import numpy as np
import pandas as pd
from datetime import timedelta
from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from .pre_train import SaveArgs, load_args


def predict(args, model,  
            predict_root = None, predict_data = None, 
            days_to_predict = 1, retrain = False, new_data = None):#model= setting or actual model
    """
    Use Model To Predict Future Days!
    Argumans:
        args: Object | str, The models setup. Can be an Object of type DotDict class, or the path to saved file of it -> (args.json).
        model: str|Object, Whether can be the setting or folder name of path to the 'checkpoint.pth' or the actual model object!
        days_to_predict: int, How much days, should to be predicted!
        predict_data: the name of predict data inside pred folder. if None, will use the current name in args.
        
        retrain: bool, Optional. If True, and new_data is not None, It would change the setting and args to retrain the current model with new data.
        new_data: str, The new data name inside the root path from args. If None, and retrain is True, It would use the current root path and data name in args to retrain model.
                \Will Raise an Error, If no data is available/
    """
    
    if isinstance(args, str):
        try:
            arg = load_args(args)
        except Exception as e:
            raise AssertionError(f"Fail to read args.pkl reason -> {e}")
    else:
        try:
            args_path = SaveArgs(args=args, path='', temporary=True)
            args_path = args_path.path
            arg = load_args(args_path)
            os.unlink(args_path)
        except Exception as e:
            raise AssertionError(f"Fail to read args.pkl reason -> {e}")
    
    
    if retrain and new_data is not None:
        arg.data_path = new_data
    
    if predict_data is not None:
        if predict_root is not None:
            arg.pred_root_path = predict_root
        arg.pred_data_path = predict_data
    
    if isinstance(model, Exp_Long_Term_Forecast) or isinstance(model, Exp_Long_Term_Forecast_Partial):
        model.args = arg
        exp = model
    elif isinstance(model, str):
        if arg.exp_name == 'partial_train':
            Exp = Exp_Long_Term_Forecast_Partial
        else:
            Exp = Exp_Long_Term_Forecast
        exp = Exp(arg)
        try:
            path = os.path.join(arg.checkpoints, model)
            path = path + '/' + 'checkpoint.pth'
            exp.model.load_state_dict(torch.load(path))
        except Exception as e:
            try:
                exp.model.load_state_dict(torch.load(model))
            except:
                raise AssertionError(f" There was an Error loading your model with the provded path.Assumed path is {model} and Error was: {e}")
    else:
        raise TypeError(" The Model Object can be of type str(model checkpoint.pth path) or the actual model from experiments kind of models from this repo.")
    
    if retrain:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for ii in range(arg.itr):
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}'.format(
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
                    arg.class_strategy, ii, timestamp)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
    
    try:
        df_temp = pd.read_csv(os.path.join(arg.pred_root_path, arg.pred_data_path))
    except:
        print(f'please inter the path to your prediction data in input arguman : predict_root  and  predict_data')
        print('Where predict_root is the main folder contained your csv file and predict_data is name of the csv file with .csv at the end')
        return 0
    end_at_first = df_temp.shape[0] - 1
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = f"{temp_file.name}.csv"
        df_temp.to_csv(temp_path, index= False)
        temp_file.seek(0)
    del df_temp
    
    folder_path = 'results/Prediction Results/'
    os.makedirs(folder_path, exist_ok=True)
    file_path = folder_path + 'prediction.csv'
    
    if os.path.exists(file_path):
        base, ext = os.path.splitext(file_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = f"{base}_{timestamp}{ext}"
    
    for jj in range(days_to_predict):
        if jj == 0:
            pass
        else:
            arg.pred_root_path = 'None'
            arg.pred_data_path = temp_path
            exp.args = arg
        pred_data, pred_loader = exp._get_data(flag='pred')
        preds = []
        exp.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(exp.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(exp.device)
                batch_y_mark = batch_y_mark.float().to(exp.device)
                dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
                if exp.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if exp.args.output_attention:
                            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if exp.args.output_attention:
                        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        #preds = [round(any_) for any_ in preds.reshape(-1).tolist()]
        preds = list(preds[0,0,:])
        data = pd.read_csv(temp_path)
        cols = list(data.columns)
        date_name = arg.name_of_col_with_date if hasattr(arg, 'name_of_col_with_date') else 'date'
        target = arg.target
        data[date_name] = pd.to_datetime(data[date_name])
        last_day = data.loc[data.shape[0]-1,date_name]
        next_day = last_day + timedelta(days=1)
        date_index = cols.index(date_name)
        cols.pop(date_index)
        temp = {}
        for i in range(len(cols)):
            col = cols[i]
            if col == target :
                if arg.features == 'MS' or arg.features == 'S' :
                    temp[col] = preds[-1]
                else:
                    temp[col] = preds[i]
            else:
                if arg.features == 'S':
                    temp[col] = data.loc[end_at_first, col]
                else:
                    temp[col] = preds[i] 
        temp = pd.DataFrame(temp, index=[data.shape[0]], dtype=int)
        temp.insert(loc = date_index, column=date_name, value=next_day)
        data = pd.concat([data, temp])
        if days_to_predict > 1:
            data.to_csv(temp_path, index = False)
            #if use_predict_on_prediction and retrain:
            #    if arg.data == 'custom':
            #        arg.root_path = 'None'
            #        arg.data_path = temp_path
            #        exp.args = arg
            #        exp.train(setting)
            #    else:
            #        print("sorry can not be done")
    
    
    if arg.features == 'S' or arg.features == 'MS':
            data = pd.concat( [data.loc[end_at_first:,date_name], data.loc[end_at_first:,target]],axis=1)
    else:
        data = data.loc[end_at_first:,:]
    data.to_csv(file_path, index = False)
    os.unlink(temp_path)
    print(f'''The Results of Prediction for The Next {days_to_predict} Days Are Now Stored in 
                {file_path}''')
    return True