import torch
import torch.nn as nn

class RevNorm(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, axis=-2, eps=1e-5, affine=True):
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.affine = affine
        self.mean = None
        self.stdev = None
        
    def build(self, input_shape):
        if self.affine and not hasattr(self, 'affine_weight'):
            self.affine_weight = nn.Parameter(torch.ones(input_shape[-1]))
            self.affine_bias = nn.Parameter(torch.zeros(input_shape[-1]))
    
    def forward(self, x, mode='norm', target_slice=None):
        # Build affine parameters if not already built
        if self.affine and not hasattr(self, 'affine_weight'):
            self.build(x.shape)
            
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
        return x
    
    def _get_statistics(self, x):
        # Calculate statistics along the specified axis
        self.mean = x.mean(dim=self.axis, keepdim=True).detach()
        self.stdev = torch.sqrt(
            x.var(dim=self.axis, keepdim=True, unbiased=False) + self.eps
        ).detach()
    
    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    
    def _denormalize(self, x, target_slice=None):
        if self.affine:
            if target_slice is not None:
                x = (x - self.affine_bias[target_slice]) / self.affine_weight[target_slice]
            else:
                x = (x - self.affine_bias) / self.affine_weight
                
        if target_slice is not None:
            x = x * self.stdev[:, :, target_slice] + self.mean[:, :, target_slice]
        else:
            x = x * self.stdev + self.mean
        return x 