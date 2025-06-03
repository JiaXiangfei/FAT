import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def FFT_sim(x):
    batch_size, n_vars, seq_len = x.shape
    x = x.permute(0, 2, 1).reshape(batch_size * n_vars, seq_len)
    fft_result = torch.fft.fft(x)
    magnitude = torch.abs(fft_result)
    freq_normalized = torch.nn.functional.normalize(magnitude, p=2, dim=1)
    sim_matrix = torch.matmul(freq_normalized.reshape(freq_normalized.shape[0], -1),
                              freq_normalized.reshape(freq_normalized.shape[0], -1).T)
    mask = torch.ones(batch_size * n_vars, batch_size * n_vars)
    b = torch.zeros(n_vars, n_vars)
    for i in range(batch_size):
        mask[i * n_vars:(i + 1) * n_vars, i * n_vars:(i + 1) * n_vars] = b
    mask = mask.to(x.device)
    sim_matrix = sim_matrix.to(x.device)
    sim_matrix = sim_matrix * mask
    return sim_matrix

def generate_CLLabels(x, k_positive, k_negative):
    """
    Generate labels of shape (N, 6) with the first 3 columns as 1s and the last 3 columns as 0s.

    Parameters:
        N (int): Number of rows in the label tensor.
    Returns:
        torch.Tensor: Tensor of shape (N, 6) with the specified pattern.
    """
    N = x.shape[0]
    # Create a tensor of ones for the first 3 columns
    ones_positive = torch.ones((N, k_positive))
    ones_negative =  torch.ones((N, k_negative))
    # Concatenate the tensors along the second dimension (columns)
    labels = torch.cat((ones_positive, 1-ones_negative), dim=1)

    return labels.to(x.device)

def adjust_learning_rate(optimizer, epoch, args, learning_rate):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))

    if args.lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
        
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, pred_len):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, pred_len)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, pred_len)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, pred_len):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + f'checkpoint_{pred_len}.pth')
        self.val_loss_min = val_loss