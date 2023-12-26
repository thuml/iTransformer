from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred
from torch.utils.data import DataLoader
# DataLoader 是一个迭代器，最基本的使用方法就是传入一个 Dataset 对象，它会根据参数 batch_size 的值生成一个 batch 的数据，节省内存的同时，它还可以实现多进程、数据打乱等处理。

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    # 通过指定flag类型，分别调用训练/测试/验证数据集
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq
    data_set = Data(
        root_path=args.root_path,#'./dataset/PEMS'
        data_path=args.data_path,#'PEMS03.npz'
        flag=flag,#train/val/test
        size=[args.seq_len, args.label_len, args.pred_len],# (96, 48, 12)
        features=args.features,#'M'
        target=args.target,#’OT‘
        timeenc=timeenc,#1
        freq=freq,#h
    )
    print(flag, len(data_set))
    # dataset表示Dataset类，它决定了数据从哪读取以及如何读取；
    # batch_size表示批大小；
    # num_works表示是否多进程读取数据；
    # shuffle表示每个epoch是否乱序；
    # drop_last表示当样本数不能被batch_size整除时，是否舍弃最后一批数据
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,#1(测试)/args.batch_size（训练）
        shuffle=shuffle_flag,#False(测试,pred)/True（训练）
        num_workers=args.num_workers,#默认10
        drop_last=drop_last)#测试，训练：True；pred:False
    return data_set, data_loader
