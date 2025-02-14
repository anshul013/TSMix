import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = 'gs://time_series_datasets'
LOCAL_CACHE_DIR = './dataset/'

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    def __init__(self, data, seq_len, pred_len, target_slice=None):
        self.data = torch.FloatTensor(data.values)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_slice = target_slice
        self.total_len = seq_len + pred_len
        
    def __len__(self):
        return len(self.data) - self.total_len + 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        if self.target_slice is not None:
            y = self.data[idx + self.seq_len:idx + self.total_len, self.target_slice]
        else:
            y = self.data[idx + self.seq_len:idx + self.total_len]
        return x, y

class TSFDataLoader:
    """Generate data loader from raw data."""
    def __init__(
        self,
        data,
        batch_size,
        seq_len,
        pred_len,
        feature_type,
        target='OT',
        num_workers=0
    ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.num_workers = num_workers
        self.target_slice = slice(0, None)
        
        self._read_data()
        
    def _read_data(self):
        """Load raw data and split datasets."""
        # Create cache directory if not exists
        if not os.path.isdir(LOCAL_CACHE_DIR):
            os.makedirs(LOCAL_CACHE_DIR)
            
        file_name = self.data + '.csv'
        cache_filepath = os.path.join(LOCAL_CACHE_DIR, file_name)
        
        # Download data if not in cache
        if not os.path.isfile(cache_filepath):
            # Note: You'll need to implement your own data download mechanism
            # The original uses tf.io.gfile.copy which isn't available in PyTorch
            raise NotImplementedError("Please implement data download mechanism")
            
        df_raw = pd.read_csv(cache_filepath)
        
        # S: univariate-univariate, M: multivariate-multivariate
        # MS: multivariate-univariate
        df = df_raw.set_index('date')
        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)
            
        # Split train/valid/test
        n = len(df)
        if self.data.startswith('ETTm'):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif self.data.startswith('ETTh'):
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n
            
        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len:val_end]
        test_df = df[val_end - self.seq_len:test_end]
        
        # Standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)
        
        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)
            
        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]
        
    def inverse_transform(self, data):
        """Inverse transform the scaled data."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        return self.scaler.inverse_transform(data)
        
    def get_train(self, shuffle=True):
        """Get training data loader."""
        dataset = TimeSeriesDataset(
            self.train_df,
            self.seq_len,
            self.pred_len,
            self.target_slice
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def get_val(self):
        """Get validation data loader."""
        dataset = TimeSeriesDataset(
            self.val_df,
            self.seq_len,
            self.pred_len,
            self.target_slice
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def get_test(self):
        """Get test data loader."""
        dataset = TimeSeriesDataset(
            self.test_df,
            self.seq_len,
            self.pred_len,
            self.target_slice
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 