import argparse
import torch
from exp.exp_fresim import Exp_fresim
import random
import numpy as np
import os


fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='FAT')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain', help='task name, options:[pretrain, finetune]')
parser.add_argument('--task_type', type=str, default='reg', help='task name, options:[pretrain, finetune]')

parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model', type=str, help='model name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./results/checkpoints/', help='location of model fine-tuning checkpoints')
parser.add_argument('--pretrain_checkpoints', type=str, default='./results/pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--transfer_checkpoints', type=str, default='ckpt_best_ptmode:{}.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')
parser.add_argument('--select_channels', type=float, default=1, help='select the rate of channels to train')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--fc_dropout', type=float, default=0, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--patch_len', type=int, default=12, help='path length')
parser.add_argument('--stride', type=int, default=12, help='stride')

# optimization
parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--pretrain_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

# Pre-train
parser.add_argument('--lm', type=int, default=3, help='average masking length')
parser.add_argument('--positive_nums', type=int, default=3, help='masking series numbers')
parser.add_argument('--negative_nums', type=int, default=1, help='masking series numbers')
parser.add_argument('--rbtp', type=int, default=1, help='0: rebuild the embedding of oral series; 1: rebuild oral series')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
parser.add_argument('--masked_rule', type=str, default='geometric', help='geometric, random, masked tail, masked head')
# parser.add_argument('--augmode', type=str, default='DistortUnmaskedPosition', help='DistortUnmaskedPosition, DistortMaskArea, Mixed')
parser.add_argument('--pretrain_mode', type=str, default='1', help='0:native_masking, 1:contrast_with_noise')
parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')
parser.add_argument('--distort', type=bool, default=True, help='whether_to_distort')
parser.add_argument('--forecastMode', type=str, default='freq', help='freq or unfreq')


parser.add_argument('--n_fft', type=int, default=128, help='Number of FFT points for STFT')
parser.add_argument('--hop_length', type=int, default=32, help='Hop length for STFT, typically n_fft // 4')
parser.add_argument('--win_length', type=int, default=128, help='Window length for STFT, usually same as n_fft')
parser.add_argument('--window', type=str, default='hann', help='Type of window function, e.g., hann')
parser.add_argument('--stft_top_k', type=int, default=5, help='Number of top elements to select')

parser.add_argument('--exp_name', type=str, default='test', help='name of exp')
parser.add_argument('--hidden_size', type=int, default=256, help='size of Pai∂ç')
parser.add_argument('--trs', type=int, default = 0, help='1: train_from_scratch, 0 : no')
parser.add_argument('--freeze', type=int, default = 0, help='1: freeze_weight, 0: no')
parser.add_argument('--n_knlg', type=int, default = 8, help='base frequency nums')

# classification
parser.add_argument('--pretrain_dataset', type=str, default='SLG', help='data file')


args = parser.parse_args()


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]

print('Args in experiment:')
print(args)

Exp = Exp_fresim

if args.task_name == 'pretrain':
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_lm{}_pn{}_mr{}_tp{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.factor,
            args.dropout,
            args.head_dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.lm,
            args.positive_nums,
            args.mask_rate,
            args.temperature
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>start pre_training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.pretrain()
        torch.cuda.empty_cache()

elif args.task_name == 'finetune':
    for ii in range(args.itr):
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_ptmode{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.factor,
            args.dropout,
            args.head_dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.pretrain_mode
        )


        args.load_checkpoints = os.path.join(args.pretrain_checkpoints, args.data, args.exp_name, args.transfer_checkpoints.format(args.pretrain_mode))

        exp = Exp(args)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test()
        torch.cuda.empty_cache()
