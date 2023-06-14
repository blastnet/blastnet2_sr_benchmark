#written by W.T. Chung

import torch 

def initialize_weights(m,activation,a,scale):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=activation,a=a)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.ConvTranspose3d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=activation,a=a)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=activation,a=a)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    else:
        print('no init for ',m)
        pass
    m.weight.data *= scale