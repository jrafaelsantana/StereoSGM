import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import cropND

from .groupnorm import GroupNorm

class SiameseOneWindow(nn.Module):
    def __init__(self, chn=1, padding_parameter=0):
        super(SiameseOneWindow, self).__init__()
        self.conv_7 = nn.Sequential(
            nn.Conv2d(chn, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(256, 128, 1, padding=0),
        )

        self.full = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self.gn = GroupNorm(1,1,0)

    def forward_one_7(self, x):
        x = self.conv_7(x)
        #x = self.gn(x)
        return x
    
    def linear (self, x):
        x = self.full(x)
        return x

    def norm (self, x):
        x = self.gn(x)
        return x

    def forward(self, x1, x2, training = True):  
        if training:
            batch_size, channel_size, width, height = x1.shape

            out1_small = self.forward_one_7(x1)
            out2_small = self.forward_one_7(x2)
            out1_small = out1_small.view(out1_small.size()[0], -1)
            out2_small = out2_small.view(out2_small.size()[0], -1)

            conc_tensor = torch.cat((out1_small, out2_small), 1)
            out = self.linear(conc_tensor)

            return out
        
        else:
            out1_small = self.forward_one_7(x1)
            out2_small = self.forward_one_7(x2)

            return out1_small, out2_small

def weights_init_uniform_rule_one_window(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


# For test
if __name__ == '__main__':
    net = SiameseOneWindow()
    print(net)
    print(list(net.parameters()))
