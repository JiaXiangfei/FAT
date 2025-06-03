
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.increased_dim = 1
        self.input_channels = 1
        self.increased_dim = 1
        self.onelayer_out_channels = 64
        self.twolayer_out_channels = 128
        self.final_out_channels = 256
        self.conv2_kernel_size = 8
        self.conv3_kernel_size = 8
        self.d_ff = 256
        self.num_classes = 5
        self.num_classes_target = 2
        self.dropout = 0.2
        self.masking_ratio = 0.5
        self.lm = 3
        self.kernel_size = 25
        self.stride = 3
        self.features_len = 127
        self.features_len_f = self.features_len
        self.TSlength_aligned = 178
        self.ft_dropout = 0.2
        self.CNNoutput_channel = 10
        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-8
        self.lr_f = self.lr

        # data parameters
        self.drop_last = True
        self.batch_size = 32

        self.target_batch_size = 32
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        self.negative_nums = 1
        self.positive_nums = 3
        self.n_knlg = 8
        self.head_dropout = 0.2
        self.hidden_dimension = 1280



class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 50
