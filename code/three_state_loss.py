import torch
from torch import nn

class ThreeStateLoss(nn.Module):
    def __init__(self, temperature_f, density_quantile=0.9):
        super().__init__()
        if not temperature_f > 0 or not 0 < density_quantile < 1:
            raise ValueError('Invalid temperature or density quantile')
        self.temperature_f = float(temperature_f)
        self.density_quantile = float(density_quantile)
        self.begin_epoch(1)

    def begin_epoch(self, num_views):
        if num_views < 1:
            raise ValueError('num_views must be positive')
        self.num_views = num_views
        self.calls = 0

    def masks(self, w, d, y_pse, density_quantile=None):
        n = w.size(0)
        if w.shape != (n, n) or d.shape != w.shape or y_pse.shape != w.shape:
            raise ValueError('Pair inputs must be equally sized square matrices')
        q = self.density_quantile if density_quantile is None else density_quantile
        if not 0 < q < 1:
            raise ValueError('density_quantile must be in (0, 1)')
        eye = torch.eye(n, dtype=torch.bool, device=w.device)
        values = d[~eye]

        threshold = (values.sort().values[min(values.numel() - 1, int(q * values.numel()))]
                     if values.numel() else d.new_zeros(()))
        adjacent = w.bool() | eye
        equal = y_pse.bool()
        positive = (adjacent & equal & (d > threshold)) | eye
        negative = (~adjacent) & (~equal) & (d <= threshold) & (~eye)
        uncertain = (~positive) & (~negative) & (~eye)
        return positive, negative, uncertain, eye
