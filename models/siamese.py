import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .groupnorm import GroupNorm

class Siamese(nn.Module):

    def __init__(self, chn=1, padding_parameter=0):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chn, 128, 3, padding=padding_parameter),  # 64@96*96
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=padding_parameter),
            nn.ReLU(),    # 128@42*42

            nn.Conv2d(128, 256, 3, padding=padding_parameter),
            nn.ReLU(),    # 128@18*18

            nn.Conv2d(256, 256, 1, padding=padding_parameter),
        )

        #self.liner = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.linear = nn.Sequential(nn.Linear(256, 1))
        self.gn = GroupNorm(1,1,0)

    def forward_one(self, x):
        x = self.conv(x)
        #x = F.normalize(x, dim=1, p=2)
        #x = x.view(x.size()[0], -1)
        #x = self.liner(x)
        return x

    def forward(self, x1, x2, training = True):
        #x1 = self.gn(x1)
        out1 = self.forward_one(x1)
        out1 = self.gn(out1)

        #x2 = self.gn(x2)
        out2 = self.forward_one(x2)
        out2 = self.gn(out2)

        if training:
            out1 = out1.view(out1.size()[0], -1)
            out2 = out2.view(out2.size()[0], -1)

            #out1 = out1.view(out1.size()[0], -1, out1.size()[1])
            #out2 = out2.view(out2.size()[0], out2.size()[1], -1)
            #out = torch.abs(torch.sum(out1 * out2, 1))
            out = torch.sqrt(torch.sum((out1 - out2) * (out1 - out2), 1))
            #out = torch.sqrt(torch.sum(torch.pow(out1 * out2, 2),1))
            #out = torch.sum((out1 - out2) * (out1 - out2), 1)

            #print(out1.shape)
            #print(out2.shape)
            #out = torch.abs(out1 - out2)
            #out = torch.abs(torch.sum((out1 - out2), 1))
            #print(out.shape)

            return out
        else:
            return out1, out2


        # print(x1.shape)
        
        
        #out1 = out1.view(out1.size()[0], -1)
        
        #qn = torch.norm(out1, p=2, dim=1, keepdim=True).detach()
        # print(qn.shape)
        # print(qn.expand_as(out1))
        # input()
        #out1 = out1.div(qn.expand_as(out1))
        #out1 = F.normalize(out1, dim=1, p=2)
        
        #out2 = out2.view(out2.size()[0], -1)
        #qn = torch.norm(out2, p=2, dim=1,keepdim=True).detach()
        #out2 = out2.div(qn.expand_as(out2))
        #out2 = F.normalize(out2, dim=1, p=2)

        # net_te = net_tr:clone('weight', 'bias')

        #out1 = out1.view(out1.size()[0], -1, out1.size()[1])
        #out2 = out2.view(out2.size()[0], out2.size()[1], -1)

        #dis = torch.abs(out1 - out2)
        
        #Anterior
        #out = torch.sum((out1 - out2) * (out1 - out2), 1)
        
        #dis = torch.cat((out1,out2),0)
        #dis = self.liner(dis)

        #out = self.out(dis)

        # print(out1.shape)


        #dis = torch.bmm(out1, out2)
        #dis = torch.abs(out1 - out2)
        #dis = torch.dot(out1,out2)
        #out = torch.sum(torch.mul(out1,out2),dim=1)
        # print(pd.shape)
        #dis = torch.sum(pd, dim=1)
        # print(dis.shape)
        #out = dis.view(dis.size()[0], -1)
        # print(dis.shape)

        #out = dis
        #return out1, out2
        #return out


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
