import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3),  # 64@96*96
            nn.ReLU(),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(),    # 128@42*42

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),    # 128@18*18

            nn.Conv2d(128, 128, 3),
            nn.ReLU(),   # 256@6*6

            nn.Conv2d(128, 128, 3),

        )

        #self.liner = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.liner = nn.Sequential(nn.Linear(128, 1))

    def forward_one(self, x):
        x = self.conv(x)
        #x = F.normalize(x, dim=1, p=2)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):

        # print(x1.shape)
        out1 = self.forward_one(x1)
        # print(out1.shape)
        #qn = torch.norm(out1, p=2, dim=1, keepdim=True).detach()
        # print(qn.shape)
        # print(qn.expand_as(out1))
        # input()
        #out1 = out1.div(qn.expand_as(out1))
        #out1 = F.normalize(out1, dim=1, p=2)

        out2 = self.forward_one(x2)
        #qn = torch.norm(out2, p=2, dim=1,keepdim=True).detach()
        #out2 = out2.div(qn.expand_as(out2))
        #out2 = F.normalize(out2, dim=1, p=2)

        # net_te = net_tr:clone('weight', 'bias')

        #out1 = out1.view(out1.size()[0], -1, out1.size()[1])
        #out2 = out2.view(out2.size()[0], out2.size()[1], -1)

        dis = torch.abs(out1 - out2)
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
        out = dis.view(dis.size()[0], -1)
        # print(dis.shape)

        #out = dis
        return out


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
