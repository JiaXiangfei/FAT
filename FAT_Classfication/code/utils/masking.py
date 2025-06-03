import torch
import numpy as np


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

def random_mask(X, masking_ratio=0.25):
    mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True, p=(1 - masking_ratio, masking_ratio))
    return torch.tensor(mask)

def geometric_mask(X, mask_rate, lm, max_length):
    if len(X.shape) == 3:
        mask = geom_noise_mask_single(X.shape[0] * X.shape[1] * X.shape[2],  mask_rate, lm, max_length)
        mask = mask.reshape(X.shape[0], X.shape[1], X.shape[2])
    elif len(X.shape) == 2:
        mask = geom_noise_mask_single(X.shape[0] * X.shape[1],  mask_rate, lm, max_length)
        mask = mask.reshape(X.shape[0], X.shape[1])
    return mask

def geom_noise_mask_single(L, masking_ratio, lm,  max_length):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]
    state = int(np.random.rand() > masking_ratio)
    keep_mask[0] = state
    continues_count = 1
    for i in range(1, L):
        keep_mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state
        if keep_mask[i] == keep_mask[i-1]:
            continues_count += 1
        else:
            continues_count = 1
        if continues_count >= max_length:
            state = 1 - state
    return keep_mask

def get_mask(x, mask_name, mask_rate, lm, max_length):
    if mask_name == 'geometric':
        mask = geometric_mask(x, mask_rate, lm, max_length)
    elif mask_name == 'random':
        mask = random_mask(x, masking_ratio=mask_rate)
    return torch.from_numpy(mask).to(x.device)
