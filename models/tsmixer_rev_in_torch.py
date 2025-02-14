import torch
import torch.nn as nn

from models.rev_in_torch import RevNorm
from models.tsmixer_torch import ResBlock

class TSMixerRevIN(nn.Module):
    """TSMixer model with Reversible Instance Normalization."""
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
        
        # RevIN layer
        self.rev_norm = RevNorm(axis=-2)
        
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
        
        # Apply RevIN normalization
        x = self.rev_norm(x, mode='norm')
        
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
        
        # Apply RevIN denormalization
        x = self.rev_norm(x, mode='denorm', target_slice=self.target_slice)
        
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
    """Build TSMixer model with RevIN."""
    return TSMixerRevIN(
        input_shape=input_shape,
        pred_len=pred_len,
        norm_type=norm_type,
        activation=activation,
        n_block=n_block,
        dropout=dropout,
        ff_dim=ff_dim,
        target_slice=target_slice
    ) 