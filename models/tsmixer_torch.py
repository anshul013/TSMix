import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual block of TSMixer."""
    def __init__(self, input_dim, ff_dim, norm_type='L', activation='relu', dropout=0.1):
        super().__init__()
        self.norm_type = norm_type
        
        # Normalization layers
        if norm_type == 'L':
            self.norm1 = nn.LayerNorm(normalized_shape=[input_dim])
            self.norm2 = nn.LayerNorm(normalized_shape=[input_dim])
        else:  # 'B' for BatchNorm
            self.norm1 = nn.BatchNorm1d(num_features=input_dim)
            self.norm2 = nn.BatchNorm1d(num_features=input_dim)
        
        # Temporal mixing
        self.temporal_mix = nn.Linear(in_features=input_dim, out_features=input_dim)
        
        # Feature mixing
        self.feature_mix1 = nn.Linear(in_features=input_dim, out_features=ff_dim)
        self.feature_mix2 = nn.Linear(in_features=ff_dim, out_features=input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Activation
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        
        # Temporal mixing
        if self.norm_type == 'L':
            normed = self.norm1(x)  # LayerNorm over features
        else:
            normed = self.norm1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm over time
            
        # Temporal linear transformation
        temporal = x.transpose(1, 2)  # [batch_size, channels, seq_len]
        temporal = self.temporal_mix(temporal)  # Mix across time dimension
        temporal = temporal.transpose(1, 2)  # [batch_size, seq_len, channels]
        temporal = self.dropout(temporal)
        temporal = temporal + x  # Residual connection
        
        # Feature mixing
        if self.norm_type == 'L':
            normed = self.norm2(temporal)
        else:
            normed = self.norm2(temporal.transpose(1, 2)).transpose(1, 2)
            
        feature = self.feature_mix1(normed)  # [batch_size, seq_len, ff_dim]
        feature = self.activation(feature)
        feature = self.dropout(feature)
        feature = self.feature_mix2(feature)  # [batch_size, seq_len, channels]
        feature = self.dropout(feature)
        
        return feature + temporal  # Residual connection

class TSMixer(nn.Module):
    """TSMixer model for time series forecasting."""
    def __init__(
        self,
        input_shape,  # (seq_len, channels)
        pred_len,
        norm_type='L',
        activation='relu',
        n_block=2,
        dropout=0.1,
        ff_dim=2048,
        target_slice=None
    ):
        super().__init__()
        self.seq_len, self.channels = input_shape
        self.pred_len = pred_len
        self.target_slice = target_slice
        
        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(
                input_dim=self.channels,
                ff_dim=ff_dim,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout
            ) for _ in range(n_block)
        ])
        
        # Final prediction layer
        self.predictor = nn.Linear(in_features=self.seq_len, out_features=pred_len)

    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        
        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply target slice if specified
        if self.target_slice is not None:
            x = x[:, :, self.target_slice]
        
        # Transpose and predict
        x = x.transpose(1, 2)  # [batch_size, channels, seq_len]
        x = self.predictor(x)  # [batch_size, channels, pred_len]
        x = x.transpose(1, 2)  # [batch_size, pred_len, channels]
        
        return x

def build_model(
    input_shape,
    pred_len,
    norm_type='L',
    activation='relu',
    n_block=2,
    dropout=0.1,
    ff_dim=2048,
    target_slice=None
):
    """Build TSMixer model."""
    return TSMixer(
        input_shape=input_shape,
        pred_len=pred_len,
        norm_type=norm_type,
        activation=activation,
        n_block=n_block,
        dropout=dropout,
        ff_dim=ff_dim,
        target_slice=target_slice
    ) 