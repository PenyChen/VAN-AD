from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from data_factory import utils
from argparse import Namespace
from sklearn.preprocessing import StandardScaler

class TABDatasetNF(Dataset):
    def __init__(self, config, mode, seq_len, pre_len, step, data_ratio=1.0):
        super(TABDatasetNF, self).__init__()
        self.config = config
        self.mode = mode
        self.seq_len = seq_len if seq_len is not None else config.seq_len
        self.pre_len = pre_len if pre_len is not None else config.pre_len
        self.step = step if step else config.step
        self.data_ratio = data_ratio
        self.load_meta()
        self.load_data()
        
    
    def load_meta(self):
        meta_file = os.path.join(self.config.data_path, "DETECT_META.csv")
        meta_df = pd.read_csv(meta_file)
        is_multivariate = meta_df.loc[meta_df["file_name"] == os.path.basename(self.config.tab_csv_file)]["if_univariate"].values[0] == False
        self.train_lens = meta_df.loc[meta_df["file_name"] == os.path.basename(self.config.tab_csv_file)]["train_lens"].values[0]
        self.file_dir = "multi_ts" if is_multivariate else "uni_ts"


    def load_data(self):
        data = utils.read_data(os.path.join(self.config.data_path, self.file_dir, os.path.basename(self.config.tab_csv_file)))
        self.scaler = StandardScaler()
        if self.mode == 'train':
            self.data = data.values[:self.train_lens,:-1]
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)
            self.label = np.zeros(self.data.shape[0])
            # Apply data ratio sampling
            if self.data_ratio < 1.0:
                num_samples = int(len(self.data) * self.data_ratio)
                self.data = self.data[:num_samples]
                self.label = self.label[:num_samples]
            self.train =self.data
            self.test_labels =self.label
        elif self.mode == 'test':
            self.scaler.fit(data.values[:self.train_lens,:-1])
            self.data = data.values[self.train_lens:,:-1]
            self.data = self.scaler.transform(self.data)
            self.label = data.values[self.train_lens:,-1:].reshape(-1)
            self.test = self.data
            self.test_labels = self.label
        elif self.mode == 'val':
            self.scaler.fit(data.values[:self.train_lens,:-1])
            self.data = data.values[self.train_lens:,:-1]
            self.data = self.scaler.transform(self.data)
            self.label = np.zeros(self.data.shape[0])
            self.val = self.data[:5000]
            self.test_labels = self.label
        print("[Dataset]:",self.mode,"data shape:",self.data.shape,"label shape:",self.label.shape)
        # means = np.mean(self.data,axis=0,keepdims=True)
        # stds = np.std(self.data,axis=0,keepdims=True)
        # self.data = (self.data - np.repeat(means,self.data.shape[0],axis=0)) / np.repeat(stds,self.data.shape[0],axis=0) + 1e-5
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


if __name__ == "__main__":
    # self.seq_len = seq_len if seq_len is not None else config.seq_len
    #     self.pre_len = pre_len if pre_len is not None else config.pre_len
    #     self.step = step if step else config.step
    #     self.data_ratio = data_ratio
    config = Namespace()
    config.tab_csv_file = "MSL.csv"
    config.seq_len = 30
    config.pre_len = 10
    config.step = 1
    dataset = TABDataset(config, mode="train", seq_len=None, pre_len=None, step=None, data_ratio=1)
    print("Dataset length:", len(dataset))
    print("First sample shapes:", dataset[0][0].shape, dataset[0][1].shape)
    dataset = TABDataset(config, mode="test", seq_len=None, pre_len=None, step=None, data_ratio=1)
    print("Dataset length:", len(dataset))
    print("First sample shapes:", dataset[0][0].shape, dataset[0][1].shape)