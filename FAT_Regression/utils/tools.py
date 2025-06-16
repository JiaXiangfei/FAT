import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 1 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            get_record(path + '/' + 'finetune_loss.txt', f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def compare_tensors(tensor1, tensor2):
    # Use PyTorch's element-wise comparison function to create a tensor of 1s and 0s
    comparison = torch.ge(tensor1, tensor2)

    # Convert the boolean tensor to a tensor of 1s and 0s
    result = comparison.int()

    return result.type_as(torch.LongTensor())

def transfer_weights(weights_path, model, exclude_head=True, device='cpu', freeze = 0):
    new_state_dict = torch.load(weights_path,  map_location=device)['model_state_dict']

    matched_layers = 0
    unmatched_layers = []

    for name, param in model.named_parameters():
        if exclude_head and 'head' in name: continue
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.detach().copy_(input_param)
                if freeze == 1:
                    param.requires_grad = False
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model

def show_series(batch_x, batch_x_m, pred_batch_x, idx, time_points=336):

    batch_x = batch_x.permute(0, 2, 1).reshape(batch_x.shape[0], -1)
    batch_x_m = batch_x_m.permute(0, 2, 1).reshape(batch_x_m.shape[0], -1)
    pred_batch_x = pred_batch_x.permute(0, 2, 1).reshape(batch_x_m.shape[0], -1)

    bs = batch_x.shape[0]

    if time_points is None:
        time_points = batch_x.shape[1]

    positive_numbers = batch_x_m.shape[0] // bs

    batch_x = batch_x.numpy()
    batch_x_m = batch_x_m.numpy()

    x = list(range(time_points))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    for t_i in range(time_points - 1):
        for pn in range(positive_numbers):
            s_i = pn * bs + idx
            if batch_x_m[s_i][t_i] == 0:
                axs[0].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], ':', color='grey', alpha=0.5, label='masked')
            else:
                axs[0].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], '-', color=colors[pn], label='unmasked')

        axs[1].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], '-', color='blue', label='original')
        axs[1].plot([x[t_i], x[t_i + 1]], [pred_batch_x[idx][t_i], pred_batch_x[idx][t_i + 1]], '-', color='orange', label='prediction')

    axs[0].set_title('Multi-masked time series')
    axs[0].set_xlabel('X - time points')
    axs[0].set_ylabel('Y - time values')

    axs[1].set_title('Original vs Reconstruction')
    axs[1].set_xlabel('X - time points')
    axs[1].set_ylabel('Y - time values')

    return fig

def show_matrix(logits, positive_matrix, rebuild_weight_matrix):

    logits = logits.cpu().numpy()
    fig_logits = plt.figure(figsize=(80, 80))
    sns.heatmap(logits, cmap='coolwarm', vmin=np.min(logits), vmax=np.max(logits), annot=False, fmt='.2f', square=False)

    positive_matrix = positive_matrix.cpu().numpy()
    fig_positive_matrix = plt.figure(figsize=(80, 80))
    sns.heatmap(positive_matrix, cmap='coolwarm', vmin=0, vmax=1, annot=False, fmt='.1f', square=False)

    rebuild_weight_matrix = rebuild_weight_matrix.cpu().numpy()
    fig_rebuild_weight_matrix = plt.figure(figsize=(100, 100))
    sns.heatmap(rebuild_weight_matrix, cmap='coolwarm', vmin=0, vmax=1, annot=False, fmt='.3f', square=False)

    return fig_logits, fig_positive_matrix, fig_rebuild_weight_matrix

class ContrastiveWeight(nn.Module):

    def __init__(self, args):
        super(ContrastiveWeight, self).__init__()
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = args.positive_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)

        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k= oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape
        # get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        # generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat((torch.ones(cur_batch_shape[0], positives.shape[-1]), torch.zeros(cur_batch_shape[0], negatives.shape[-1])), dim=-1).to(batch_emb_om.device).float()

        # multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)

        return loss, similarity_matrix, logits, positives_mask


class AggregationRebuild(torch.nn.Module):

    def __init__(self, args):
        super(AggregationRebuild, self).__init__()
        self.args = args
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.positive_nums = args.positive_nums

    def forward(self, similarity_matrix, batch_emb_om):

        cur_batch_shape = batch_emb_om.shape

        # get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix /= self.temperature

        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(similarity_matrix.device).float() * 1e12
        rebuild_weight_matrix = self.softmax(similarity_matrix)

        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)

        # generate the rebuilt batch embedding (oral, others, oral's masks, others' masks)
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)

        # get oral' rebuilt batch embedding
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb



def STFT(x, n_fft, hop_length, win_length, window, top_k, k):
    batch_size, seq_len, n_vars = x.shape
    x = x.permute(0, 2, 1).reshape(batch_size * n_vars, seq_len)
    # Perform STFT
    stft_result = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    # Calculate magnitudes
    magnitude = torch.abs(stft_result)
    
    freq_normalized = torch.nn.functional.normalize(magnitude, p=2, dim=1)
    
    sim_matrix = torch.matmul(freq_normalized.reshape(freq_normalized.shape[0],-1), freq_normalized.reshape(freq_normalized.shape[0],-1).T)
    mask = torch.ones(batch_size * n_vars, batch_size * n_vars)
    b = torch.zeros(n_vars, n_vars)

    for i in range(batch_size):
        mask[i * n_vars:(i + 1) * n_vars, i * n_vars:(i + 1) * n_vars] = b
    mask = mask.to(x.device)
    sim_matrix = sim_matrix.to(x.device)
    sim_matrix = sim_matrix * mask

    fft_mask = torch.zeros_like(magnitude)
    # Filter to keep only top k frequencies in each time window
    top_k_indices = torch.topk(magnitude, top_k, dim=1).indices
    fft_mask.scatter_(1, top_k_indices, 1)
    # Apply mask to retain only top k frequencies
    filtered_stft = stft_result * fft_mask
    # Inverse STFT
    freq = torch.istft(filtered_stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=x.shape[-1])
    res = x - freq
    freq = freq.reshape(batch_size, n_vars, seq_len).permute(0, 2, 1)
    res = res.reshape(batch_size, n_vars, seq_len).permute(0, 2, 1)
    return freq, res, sim_matrix


class ContrastiveWeight_HN(nn.Module):

    def __init__(self, args):
        super(ContrastiveWeight_HN, self).__init__()
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = args.positive_nums


    def forward(self, batch_emb_om, negative_indexes, input_shape):
        #4 * 16 * 7， 128
        bs, seq_len, n_vars, k_positive,k_negative = input_shape
        # input_shape (bs, seq_len, n_vars)
        # get positives and negatives similarity
        norm_emb = F.normalize(batch_emb_om, dim=1)
        x_origin = norm_emb[:bs * n_vars]
        x_positive = norm_emb[bs * n_vars:].reshape(bs * n_vars, k_positive, -1)
        positive_similarity_matrix = torch.matmul(x_origin.unsqueeze(1), x_positive.permute(0, 2, 1)).squeeze()
        negative_similarity_matrix = torch.matmul(x_origin.unsqueeze(1), x_origin[negative_indexes].permute(0, 2, 1)).squeeze()
        if k_negative == 1:
            negative_similarity_matrix = torch.matmul(x_origin.unsqueeze(1), x_origin[negative_indexes].permute(0, 2, 1)).squeeze().unsqueeze(-1)
        else:
            negative_similarity_matrix = torch.matmul(x_origin.unsqueeze(1), x_origin[negative_indexes].permute(0, 2, 1)).squeeze()
        similarity_matrix = torch.cat([positive_similarity_matrix, negative_similarity_matrix], dim=-1)
        # generate predict and target probability distributions matrix
        y_true = generate_CLLabels(x_origin, k_positive, k_negative)
        # multiple positives - KL divergence
        predict = self.log_softmax(similarity_matrix / self.temperature)
        loss = self.kl(predict, y_true)

        return loss, similarity_matrix

class AggregationRebuild_HN(torch.nn.Module):

    def __init__(self, args):
        super(AggregationRebuild_HN, self).__init__()
        self.args = args
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.positive_nums = args.positive_nums

    def forward(self, similarity_matrix, p_enc_out, negative_index, input_shape):
        bs, seq_len, n_vars, k_positive,k_negative = input_shape
        # get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix /= self.temperature
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        rebuild_batch_emb = (torch.matmul(rebuild_weight_matrix[:, :k_positive].unsqueeze(1),
                                          p_enc_out[bs * n_vars:].reshape(bs * n_vars, k_positive, -1)) +
                             torch.matmul(rebuild_weight_matrix[:, k_positive:].unsqueeze(1),
                                          p_enc_out[:bs * n_vars][negative_index].reshape(bs * n_vars, k_negative, -1)))
        # get oral' rebuilt batch embedding
        rebuild_oral_batch_emb = torch.reshape(rebuild_batch_emb, (bs * n_vars, p_enc_out.shape[-2], -1))
        return rebuild_oral_batch_emb


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


def FFT_sim(x):
    batch_size, n_vars, seq_len = x.shape
    x = x.permute(0, 2, 1).reshape(batch_size * n_vars, seq_len)
    # Perform FFT
    fft_result = torch.fft.fft(x)
    # Calculate magnitudes
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


def get_record(path, content):
    try:
        with open(path, 'a', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        print(f"Error: {e}")
