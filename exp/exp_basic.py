import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, SOFTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SimpleNet,\
    Transformer2, SMamba, FAN, CARD, CCM_dlinear, CCM_patchtst, DUET, PaiFilter, CycleNet, Leddam, ModernTCN, \
    Image, Freq, PatchTSMixer, Pathformer, SimpleTSF, Fredformer, Channel_conv, Trans, RNN, CNN, SparseTSF, FITS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            'SimpleNet': SimpleNet,
            'Transformer2': Transformer2,
            'Image':Image,
            'Freq':Freq,
            'Trans':Trans,
            'RNN':RNN,
            'CNN':CNN,
            'SparseTSF':SparseTSF,
            'SOFTS':SOFTS,
            'SMamba':SMamba,
            'Pathformer':Pathformer,
            'ModernTCN':ModernTCN,
            'FAN':FAN,
            'CycleNet':CycleNet,
            'FITS':FITS,
            'Leddam':Leddam,
            'CARD':CARD,
            'SimpleTSF':SimpleTSF,
            'PaiFilter':PaiFilter,
            'DUET':DUET,
            'PatchTSMixer':PatchTSMixer,
            'Fredformer':Fredformer,
            'CCM_patchtst':CCM_patchtst,
            'CCM_dlinear':CCM_dlinear,
            'Channel_conv':Channel_conv
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict[Mamba] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
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
