from operator import mod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

from utils import cropND

from .groupnorm import GroupNorm

class Siamese(nn.Module):
    def __init__(self, chn=1, padding_parameter=0):
        super(Siamese, self).__init__()
        self.conv_7 = nn.Sequential(
            nn.Conv2d(chn, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 128, 1, padding=padding_parameter),
        )

        self.conv_15 = nn.Sequential(
            nn.Conv2d(chn, 32, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Dropout2d(0.3),

            nn.Conv2d(64, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 128, 1, padding=padding_parameter),
        )

        self.full = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.gn = GroupNorm(1,1,0)

    def forward_one_7(self, x):
        x = self.conv_7(x)
        x = self.gn(x)
        return x

    def forward_one_15(self, x):
        x = self.conv_15(x)
        x = self.gn(x)
        return x
    
    def linear (self, x):
        x = self.full(x)
        return x

    def forward(self, x1, x2, training = True):  
        out1 = self.forward_one_15(x1)
        out2 = self.forward_one_15(x2)

        if training:
            out1 = out1.view(out1.size()[0], -1)
            out2 = out2.view(out2.size()[0], -1)

            #x1_small = nn.functional.interpolate(x1, scale_factor=0.5, mode='bilinear')
            #x2_small = nn.functional.interpolate(x2, scale_factor=0.5, mode='bilinear')

            resize = torchvision.transforms.Resize((7,7))

            x1_small = resize(x1)
            x2_small = resize(x2)

            out1_small = self.forward_one_7(x1_small)
            out2_small = self.forward_one_7(x2_small)

            out1_small = out1_small.view(out1_small.size()[0], -1)
            out2_small = out2_small.view(out2_small.size()[0], -1)

            conc_tensor = torch.cat((out1, out1_small, out2, out2_small), 1)
            out = self.linear(conc_tensor)

            return out
        
        else:
            out1_small = self.forward_one_7(x1)
            out2_small = self.forward_one_7(x2)

            return out1_small, out2_small, out1, out2

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
