import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import cropND

from .groupnorm import GroupNorm

class Siamese(nn.Module):
    def __init__(self, chn=1, padding_parameter=0):
        super(Siamese, self).__init__()
        self.conv_7 = nn.Sequential(
            nn.Conv2d(chn, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(256, 128, 1, padding=padding_parameter),
        )

        self.conv_15 = nn.Sequential(
            nn.Conv2d(chn, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(256, 128, 1, padding=padding_parameter),
        )

        self.full = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.gn = GroupNorm(1,1,0)

    def forward_one_7(self, x):
        x = self.conv_7(x)
        #x = self.gn(x)
        return x

    def forward_one_15(self, x):
        x = self.conv_15(x)
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
            width_small = int(width/2)
            height_small = int(height/2)

            x1_small = cropND(x1, (batch_size, channel_size, width_small, height_small))
            x2_small = cropND(x2, (batch_size, channel_size, width_small, height_small))

            x1_down = nn.functional.interpolate(x1, scale_factor=0.5, mode='bilinear')
            x2_down = nn.functional.interpolate(x2, scale_factor=0.5, mode='bilinear')

            out1 = self.forward_one_15(x1_down)
            out2 = self.forward_one_15(x2_down)
            out1 = out1.view(out1.size()[0], -1)
            out2 = out2.view(out2.size()[0], -1)

            out1_small = self.forward_one_7(x1_small)
            out2_small = self.forward_one_7(x2_small)
            out1_small = out1_small.view(out1_small.size()[0], -1)
            out2_small = out2_small.view(out2_small.size()[0], -1)

            #calc1 = out1_small + out2_small
            #calc2 = out1 + out2

            #calc1 = torch.einsum('ij, ij -> ij', [out1, out2])
            #calc2 = torch.einsum('ij, ij -> ij', [out1_small, out2_small])
            #out = torch.sqrt(torch.sum((calc1 - calc2) * (calc1 - calc2), 1))
            #out = torch.sqrt(torch.sum((out1_small - out2_small) * (out1_small - out2_small), 1))
            #out = torch.abs(torch.sum(calc1 + calc2, 1))

            conc_tensor = torch.cat((out1, out1_small, out2, out2_small), 1)
            #conc_tensor = torch.cat((out1_small, out2_small), 1)
            #print(conc_tensor.shape)
            #conc_tensor = self.norm(conc_tensor)
            #conc_tensor = torch.abs(out1_small-out2_small)
            out = self.linear(conc_tensor)
            #out = torch.pow(out, 2)

            return out
        
        else:
            x1_down = nn.functional.interpolate(x1, scale_factor=0.5, mode='bilinear')
            x2_down = nn.functional.interpolate(x2, scale_factor=0.5, mode='bilinear')

            out1 = self.forward_one_15(x1_down)
            out2 = self.forward_one_15(x2_down)

            batch_size, channel_size, width, height = x1.shape
            width_small = int(width/2)
            height_small = int(height/2)

            x1_small = cropND(x1, (batch_size, channel_size, width_small, height_small))
            x2_small = cropND(x2, (batch_size, channel_size, width_small, height_small))

            out1_small = self.forward_one_7(x1_small)
            out2_small = self.forward_one_7(x2_small)

            return out1, out1_small, out2, out2_small

def weights_init_uniform_rule(m):
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
    net = Siamese()
    print(net)
    print(list(net.parameters()))
