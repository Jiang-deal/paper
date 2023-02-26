# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:30:03 2018

@author: Zhiyong
"""

from TrainModel import *

def PrepareDataset(speed_matrix, BATCH_SIZE = 40, seq_len = 10, pred_len = 1, train_propotion = 0.7, valid_propotion = 0.2):
    """ Prepare training and testing datasets and dataloaders.
    
    Convert speed/volume/occupancy matrix to training and testing dataset. 
    The vertical axis of speed_matrix is the time axis and the horizontal axis 
    is the spatial axis.
    
    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    time_len = speed_matrix.shape[0]#每个位置，不同时间点收集的数据，间隔时间为5分钟；具体值为105120
    
    speed_matrix = speed_matrix.clip(0, 100)#将其中的值，变换为0-100之间
    
    max_speed = speed_matrix.max().max()
    speed_matrix =  speed_matrix / max_speed
    
    speed_sequences, speed_labels = [], []
    # 一个LSTM的输入为10*323，输出为1*323。323指的应该是路口数量
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i+seq_len].values)
        speed_labels.append(speed_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)#转换数据类型为ndarray
    
    # shuffle and split the dataset to training and testing datasets
    sample_size = speed_sequences.shape[0]#样本个数
    index = np.arange(sample_size, dtype = int)
    np.random.shuffle(index)
    # np.floor--向下取整
    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))
    
    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    
    return train_dataloader, valid_dataloader, test_dataloader, max_speed

if __name__ == "__main__":
    
    data = 'loop'
    # INRIX数据集，还未下载到
    if data == 'inrix':
        speed_matrix =  pd.read_pickle('../../../Data_Warehouse/Data_network_traffic/inrix_seattle_speed_matrix_2012')
    #四条高速公路的数据，下载到了
    elif data == 'loop':
        speed_matrix =  pd.read_pickle('./../Data_Warehouse/Data_network_traffic/speed_matrix_2015')
    
    train_dataloader, valid_dataloader, test_dataloader, max_speed = PrepareDataset(speed_matrix)
    
    lstm, lstm_loss = TrainLSTM(train_dataloader, valid_dataloader, num_epochs = 10)
    
#    bilstm, bilstm_loss = Train_BiLSTM(train_dataloader, valid_dataloader, num_epochs = 10)
    
    # multibilstm, multibilstm_loss = Train_Multi_Bi_LSTM(train_dataloader, valid_dataloader, num_epochs = 10)