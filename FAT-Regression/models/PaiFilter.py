import torch
import torch.nn as nn
from utils.tools import ContrastiveWeight, AggregationRebuild
from utils.losses import AutomaticWeightedLoss
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, mask = None):
        if mode == 'norm':
            if mask:
                self.means = torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1).unsqueeze(1).detach()
                x = x - self.means
                x = x.masked_fill(mask == 0, 0)
                self.stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
                x /= self.stdev
                if self.affine:
                    x = x * self.affine_weight
                    x = x + self.affine_bias
                    x = x.masked_fill(mask == 0, 0)
            else:
                self._get_statistics(x)
                x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer_encoder = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size

        self.w_encoder = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc_cl = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(configs.head_dropout),
        )

        self.fc_pt_encoder = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

        self.awl = AutomaticWeightedLoss(2)
        self.contrastive = ContrastiveWeight(self.configs)
        self.aggregation = AggregationRebuild(self.configs)
        self.mse = torch.nn.MSELoss()


    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def pretrain(self, x_enc, x_mark_enc, batch_x, mask):

        # data shape
        bs, seq_len, n_vars = x_enc.shape
        z = x_enc
        z = self.revin_layer_encoder(z, 'norm')
        x = z
        x = x.permute(0, 2, 1)
        p_enc_out = self.circular_convolution(x, self.w_encoder.to(x.device))  # B, N, D
        s_enc_out = self.fc_pt_encoder(x)
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out.reshape(bs * n_vars, -1))
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out.reshape(bs * n_vars, -1))  # agg_enc_out: [(bs * n_vars) x seq_len x d_model] # buxyao

        x = agg_enc_out.reshape(bs, n_vars, -1)  # agg_enc_out: [bs x n_vars x seq_len x d_model] #
        x = self.fc_pt_encoder(x)
        x = x.permute(0, 2, 1)
        z = x
        z = self.revin_layer_encoder(z, 'denorm')
        pred_batch_x = z
        # series reconstruction #
        loss_rb = self.mse(pred_batch_x[:batch_x.shape[0]], batch_x.detach()) #
        # loss
        loss = self.awl(loss_cl, loss_rb)

        return loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x


    def forecast(self, x):
        z = x
        z = self.revin_layer_encoder(z, 'norm')
        x = z

        x = x.permute(0, 2, 1)

        x = self.circular_convolution(x, self.w_encoder.to(x.device))  # B, N, D

        x = self.fc_pt_encoder(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer_encoder(z, 'denorm')
        x = z
        return x

    def forward(self, x_enc, x_mark_enc, batch_x=None, mask=None):

        if self.task_name == 'pretrain':
            return self.pretrain(x_enc, x_mark_enc, batch_x, mask)

        if self.task_name == 'finetune':
            dec_out = self.forecast(x_enc)
            return dec_out

        return None
