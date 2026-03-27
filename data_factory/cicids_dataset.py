from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
class CICIDSDatasetNF(Dataset):
    def __init__(self, config, mode, seq_len, pre_len, step, data_ratio=1.0):
        super(CICIDSDatasetNF, self).__init__()
        self.config = config
        self.mode = mode
        self.scaler = StandardScaler()
        self.seq_len = seq_len if seq_len is not None else config.seq_len
        self.pre_len = pre_len if pre_len is not None else config.pred_len
        self.step = step if step else config.step
        self.data_ratio = data_ratio
        data = np.load(os.path.join(self.config.data_path,"CICIDS.npy"),allow_pickle=True)
        data = np.nan_to_num(data)
        self.scaler.fit(data[:85115,1:])
        train = self.scaler.transform(data)
        if data_ratio < 1.0:
            train = train[:int(len(train)*data_ratio)]

        test_data = data[85115:,1:]
        self.test = self.scaler.transform(test_data)
        self.train = train
        self.val = data[:5000]
        self.test_labels = np.load(os.path.join(self.config.data_path,"CICIDS_label.npy"),allow_pickle=True)[85115:,1:]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def load_data(self):
        pass

    
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
        