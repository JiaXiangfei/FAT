import numpy as np
import torch
import math
from utils.masking import get_mask

def augment_positive(sample, masking_ratio, lm, distribution='geometric', scale=0.1, k = 3):
    b_n, seq_len = sample.shape
    sample = sample.repeat(int(k/3), 1, 1)
    # mask
    blank_mask = torch.tensor(
        get_mask(sample, distribution, masking_ratio, lm, seq_len),
        dtype=sample.dtype,
        device=sample.device
    )

    # blank_mask
    x_blank_masked = blank_mask * sample

    # noise_mask
    t_noise = torch.tensor(
        np.random.standard_t(df=2, size=sample.shape), dtype=sample.dtype, device=sample.device
    ) * scale
    x_t_distorted = sample + t_noise * (1 - blank_mask)

    # zoom_mask
    scale_factor = torch.rand(sample.shape, dtype=sample.dtype, device=sample.device) * 1.5 + 0.5
    x_zoomed = sample * blank_mask + sample * (1 - blank_mask) * scale_factor

    x_augmented = torch.cat([x_blank_masked, x_t_distorted, x_zoomed], dim=0)
    return x_augmented.permute(1, 0, 2)