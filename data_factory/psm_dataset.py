from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
    
class PSMDatasetNF(Dataset):
    def __init__(self, config, mode, seq_len, pre_len, step, data_ratio=1.0):
        super(PSMDatasetNF, self).__init__()
        self.config = config
        self.mode = mode
        self.scaler = StandardScaler()
        self.seq_len = seq_len if seq_len is not None else config.seq_len
        self.pre_len = pre_len if pre_len is not None else config.pre_len
        self.step = step if step else config.step
        self.data_ratio = data_ratio
        data = pd.read_csv(os.path.join(self.config.data_path,"train.csv")).values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        if data_ratio < 1.0:
            data = data[:int(len(data)*data_ratio)]

        test_data = pd.read_csv(os.path.join(self.config.data_path,"test.csv")).values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data[:int(len(data)*0.7)]
        self.val = data[int(len(data)*0.7):]
        self.test_labels = pd.read_csv(os.path.join(self.config.data_path,"test_label.csv")).values[:, 1:]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def load_data(self):

        if self.mode == 'train':
            self.data = pd.read_csv(os.path.join(self.config.data_path,"train.csv")).values[:, 1:]
            self.data = np.nan_to_num(self.data)
            self.label = np.zeros(self.data.shape[0], dtype=np.float32)
            # Apply data ratio sampling
            if self.data_ratio < 1.0:
                num_samples = int(len(self.data) * self.data_ratio)
                self.data = self.data[:num_samples]
                self.label = self.label[:num_samples]
        elif self.mode == 'test':
            self.data = pd.read_csv(os.path.join(self.config.data_path,"test.csv")).values[:, 1:]
            self.data = np.nan_to_num(self.data)
            self.label = pd.read_csv(os.path.join(self.config.data_path,"test_label.csv")).values[:, 1:]   
        elif self.mode == 'val':
            self.data = pd.read_csv(os.path.join(self.config.data_path,"test.csv")).values[:, 1:]
            self.data = np.nan_to_num(self.data)
            self.label = pd.read_csv(os.path.join(self.config.data_path,"test_label.csv")).values[:, 1:] 
        return
    
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.seq_len) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.seq_len) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.seq_len) // self.step + 1
        else:
            return (self.test.shape[0] - self.seq_len) // self.seq_len + 1
    
    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.seq_len]), np.float32(self.test_labels[0:self.seq_len])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.seq_len]), np.float32(self.test_labels[0:self.seq_len])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.seq_len]), np.float32(self.test_labels[index:index + self.seq_len])
        