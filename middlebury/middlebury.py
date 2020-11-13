import numpy as np
import torch
import os.path

from torch._C import dtype

def fromfile(fname):
    file = open(fname + '.dim', 'r')
    dim = []

    for line in file.readlines():
        dim.append(int(line))

    if len(dim) == 1 and dim[0] == 0:
        return torch.Tensor()

    file = open(fname + '.type', 'r')
    typeFile = file.read()

    if typeFile == 'float32':
        x = torch.FloatTensor(np.fromfile(fname, dtype=np.float32))
    elif typeFile == 'int32':
        x = torch.IntTensor(np.fromfile(fname, dtype=np.int32))
    elif typeFile == 'int64':
        x = torch.LongTensor(np.fromfile(fname, dtype=np.int64))
    else:
        print(fname, typeFile)
        assert(False)

    x = x.reshape(dim)

    if typeFile == 'float32' and len(x.size()) == 4:
        x = x.unsqueeze(0)

    return x

def load (rect, color):
    data_dir = 'middlebury/data.mb.%s_%s' % (rect, color)
    #te = fromfile('%s/te.bin' % data_dir, device)
    metadata = fromfile('%s/meta.bin' % data_dir)
    nnz_tr = fromfile('%s/nnz_tr.bin' % data_dir)
    #nnz_te = fromfile('%s/nnz_te.bin' % data_dir, device)

    fname_submit = []

    for line in open('%s/fname_submit.txt' % data_dir, 'r').readlines():
        fname_submit.append(line.rstrip())
    
    X = []
    dispnoc = []

    for n in range(0, list(metadata.size())[0]):
        XX = []
        light = 1

        while True:
            fname = '%s/x_%d_%d.bin' % (data_dir, n+1, light)

            if not os.path.exists(fname):
                break

            XX.append(fromfile(fname))
            light = light + 1

        X.append(XX)

        fname = '%s/dispnoc%d.bin' % (data_dir, n+1)

        if os.path.exists(fname):
            dispnoc.append(fromfile(fname))

    #return X, te, metadata, nnz_tr, nnz_te
    return X, metadata, nnz_tr

    '''perm = torch.randperm(nnz.size()[0])

    for t in range(0, nnz.size()[0] - int(BATCH_SIZE/2), int(BATCH_SIZE/2)):
        for i in range(0, int(BATCH_SIZE/2)):
            ind = perm[t + i]
            img = int(nnz[ind, 0].item())
            dim3 = int(nnz[ind, 1].item())
            dim4 = int(nnz[ind, 2].item())
            d = nnz[ind, 3].item()

            lenImg = len(X[img])    # Light qty

            if lenImg:
                light = (randint(0,10000) % lenImg)

                lenExp = X[img][light].shape[0]    # Exp qty
                exp = (randint(1,10000) % lenExp)

                aux = X[img][light][exp].shape[0]

                light_ = light
                exp_ = exp

                if aux > 1:    
                    if random() < 0.2:
                        exp_ = (randint(1,10000) % lenExp)
                    
                    if random() < 0.2:
                        light_ = max(1, (randint(1,10000) % lenImg))

                    x0 = X[img][light][exp,0]
                    x1 = X[img][light_][exp_,1]
    '''
