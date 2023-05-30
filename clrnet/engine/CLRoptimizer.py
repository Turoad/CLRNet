import torch

def build_optimizer(cfg, net):
    params = []
    cfg_cp = cfg.optimizer.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type != 'Adam':
        raise ValueError("{} optimizer is not supported. Only Adam optimizer is currently supported.".format(cfg_type))

    return torch.optim.Adam(net.parameters(), **cfg_cp)
