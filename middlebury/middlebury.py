import math
import numpy as np
import torch
import os.path

BATCH_SIZE = 32

def fromfile(fname):
    file = open(fname + '.dim', 'r')
    dim = []

    for line in file.readlines():
        dim.append(int(line))

    if len(dim) == 1 and dim[0] == 0:
        return torch.Tensor()

    file = open(fname + '.type', 'r')
    typeFile = file.read()

    size = np.prod(dim)

    if typeFile == 'float32':
        x = torch.FloatTensor(torch.FloatStorage().from_file(fname, size=size))
    elif typeFile == 'int32':
        x = torch.IntTensor(torch.IntStorage().from_file(fname, size=size))
    elif typeFile == 'int64':
        x = torch.LongTensor(torch.LongStorage().from_file(fname, size=size))
    else:
        print(fname, typeFile)
        assert(False)

    x = x.reshape(dim)

    return x

if __name__ == "__main__":
    rect = "imperfect"
    color = "gray"

    data_dir = 'data.mb.%s_%s' % (rect, color)
    te = fromfile('%s/te.bin' % data_dir)
    metadata = fromfile('%s/meta.bin' % data_dir)
    nnz_tr = fromfile('%s/nnz_tr.bin' % data_dir)
    nnz_te = fromfile('%s/nnz_te.bin' % data_dir)

    fname_submit = []

    for line in open('%s/fname_submit.txt' % data_dir, 'r').readlines():
        fname_submit.append(line.rstrip())
    
    X = []
    dispnoc = []

    for n in range(1, list(metadata.size())[0]):
        XX = []
        light = 1

        while True:
            fname = '%s/x_%d_%d.bin' % (data_dir, n, light)

            if not os.path.exists(fname):
                break

            XX.append(fromfile(fname))
            light = light + 1

        X.append(XX)

        fname = '%s/dispnoc%d.bin' % (data_dir, n)

        if os.path.exists(fname):
            dispnoc.append(fromfile(fname))

    # print(te.shape)
    # print(metadata.shape)
    # print(nnz_tr.shape)
    # print(nnz_te.shape)

    nnz = nnz_tr

    perm = torch.randperm(nnz.size()[0])

    for t in range(1, nnz.size()[0] - int(BATCH_SIZE/2), int(BATCH_SIZE/2)):
        for i in range(1,int(BATCH_SIZE/2)):
            
            ind = perm[t + i - 1]
            img = int(nnz[ind, 0].item())
            dim3 = int(nnz[ind, 1].item())
            dim4 = int(nnz[ind, 2].item())
            d = nnz[ind, 3].item()

            light = (torch.randint(low=1, high=10, size=(1,1)).item() % (len(X[img])-1)) + 2
            exp = (torch.randint(low=1, high=10, size=(1,1)).item() % (X[img][light].size()[0])) + 1
            light_ = light
            exp_ = exp
            
            if torch.rand(1).item() < 0.2:
                exp_ = (torch.randint(low=1, high=10, size=(1,1)).item() % X[img][light].size()[0]) + 1
            
            if torch.rand(1).item() < 0.2:
                light_ = max(2, light - 1)

            x0 = X[img][light-1][exp-1,0]
            x1 = X[img][light_-1][exp_-1,1]

            # print(ind)
            # print(img)
            # print(dim3)
            # print(dim4)
            # print(d)
            input()

