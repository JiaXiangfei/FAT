import os
import torch
from models import SimMTM, SimMTM_HN, PaiFilter,PaiFilterOrg,FreSim, PatchTST, kbRAD, AFT, AFT_no_adapt, AFT_no_constract, AFT_no_guide, AFT_no_infer, AFT_new_cl, FAT, FAT_unrevised

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {'SimMTM': SimMTM,
                           'SimMTM_HN': SimMTM_HN,
                           'PaiFilter': PaiFilter,
                           'PaiFilterOrg':PaiFilterOrg,
                           'FreSim':FreSim,
                           'PatchTST': PatchTST,
                           'KbRAD': kbRAD,
                           'AFT':AFT,
                           'AFT_no_adapt':AFT_no_adapt,
                           'AFT_no_constract':AFT_no_constract,
                           'AFT_no_guide':AFT_no_guide,
                           'AFT_no_infer':AFT_no_infer,
                           'AFT_new_cl': AFT_new_cl,
                           'FAT': FAT,
                           'FAT_unrevised': FAT_unrevised
                           }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
