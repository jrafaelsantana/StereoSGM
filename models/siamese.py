import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils
from .groupnorm import GroupNorm
from utils import cropND

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

            nn.Conv2d(128, 128, 1, padding=padding_parameter)
        )

        self.conv_15 = nn.Sequential(
            nn.Conv2d(chn, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=padding_parameter),
            nn.ReLU(),

            nn.Conv2d(128, 128, 1, padding=padding_parameter)
        )

        self.gn = GroupNorm(1,1,0)

    def forward_one_7(self, x):
        x = self.conv_7(x)
        return x

    def forward_one_15(self, x):
        x = self.conv_15(x)
        return x

    def forward(self, x1, x2, training = True):  
        out1 = self.forward_one_15(x1)
        out1 = self.gn(out1)

        out2 = self.forward_one_15(x2)
        out2 = self.gn(out2)

        if training:
            out1 = out1.view(out1.size()[0], -1)
            out2 = out2.view(out2.size()[0], -1)

            batch_size, channel_size, width, height = x1.shape
            width_small = int(width/2)
            height_small = int(height/2)
                
            x1_small = utils.cropND(x1, (batch_size, channel_size, width_small, height_small))
            x2_small = utils.cropND(x2, (batch_size, channel_size, width_small, height_small))

            out1_small = self.forward_one_7(x1_small)
            out1_small = self.gn(out1_small)

            out2_small = self.forward_one_7(x2_small)
            out2_small = self.gn(out2_small)

            out1_small = out1_small.view(out1_small.size()[0], -1)
            out2_small = out2_small.view(out2_small.size()[0], -1)

            out = torch.sqrt(torch.sum((out1 - out2) * (out1 - out2), 1))
            out_small = torch.sqrt(torch.sum((out1_small - out2_small) * (out1_small - out2_small), 1))

            #calc = out + out_small

            # print(out1.shape)
            # print(out2.shape)

            # calc = torch.dot(out1, out2)
            # calc = torch.dot(calc, out1_small)
            # calc = torch.dot(calc, out2_small)

            # print(calc.shape)
            # input()

            # calc = torch.sum(out1 * out2 * out1_small * out2_small, 1)
            #calc1 = torch.sum(out1 * out2, 1)
            #calc2 = torch.sum(out1_small * out2_small, 1)
            #calc = calc1 + calc2

            # print(calc1.shape)
            # input()

            # out = torch.sum(calc1 + calc2, 
            # print(out.shape)
            # input()

            return out + out_small
        else:
            out1_small = self.forward_one_7(x1)
            out1_small = self.gn(out1_small)

            out2_small = self.forward_one_7(x2)
            out2_small = self.gn(out2_small)

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
