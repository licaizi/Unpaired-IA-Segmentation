import torch.nn as nn
import torch

class DSBN(nn.Module):

    def __init__(self,num_domain,num_features,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True):
        super(DSBN, self).__init__()
        self.BNs = nn.ModuleList(
            [nn.BatchNorm3d(num_features,eps,momentum,affine,track_running_stats) for i in range(num_domain)]
        )
    def forward(self,x,modal_class):
        return self.BNs[modal_class](x)
