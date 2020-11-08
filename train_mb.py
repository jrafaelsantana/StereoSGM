import random
import models
import math
import torch
import time
import numpy as np
import os
import config
import models
import utils
import middlebury

settings = config.Config()

torch.manual_seed(int(settings.seed))

USE_CUDA = int(settings.use_cuda)
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
PATCH_WIDTH = int(settings.patch_width)
PATCH_HEIGHT = int(settings.patch_height)
DH = int(settings.height_stride)
DW = int(settings.width_stride)
CENTER_PATCH_WIDTH = int(PATCH_WIDTH/2)
CENTER_PATCH_HEIGHT = int(PATCH_HEIGHT/2)
QTY_CORRECT_TRAIN = int(settings.train_correct)
QTY_INCORRECT_TRAIN = int(settings.train_incorrect)
CHANNEL_NUMBER = int(settings.channel_number)
EPOCHS_NUMBER = int(settings.epochs_number)
BATCH_SIZE = int(settings.batch_size)

#Augumentation
HSCALE = float(settings.HSCALE)
SCALE = float(settings.SCALE)
HFLIP = int(settings.HFLIP)
VFLIP = int(settings.VFLIP)
HSHEAR = float(settings.HSHEAR)
TRANS = float(settings.TRANS)
ROTATE = float(settings.ROTATE)
BRIGHTNESS = float(settings.BRIGHTNESS)
CONTRAST = float(settings.CONTRAST)
D_CONTRAST = float(settings.D_CONTRAST)
D_HSCALE = float(settings.D_HSCALE)
D_HSHEAR = float(settings.D_HSHEAR)
D_VTRANS = float(settings.D_VTRANS)
D_ROTATE = float(settings.D_ROTATE)
D_BRIGHTNESS = float(settings.D_BRIGHTNESS)
D_EXP = float(settings.D_EXP)
D_LIGHT = float(settings.D_LIGHT)

if not os.path.exists('weights/'):
    os.mkdir('weights/')

weight_path = 'weights/trainedweight.pth'

def train(batch_size, epochs_number, device, dataset_neg_low=2.5, dataset_neg_high=6.0, dataset_pos=0.5, patch_height=11, patch_width=11, channel_number=1):
    
    net = models.Siamese(channel_number).to(device)
    criterion = torch.nn.BCELoss().to(device)

    if(weight_path != None and os.path.exists(weight_path)):
        net.load_state_dict(torch.load(weight_path))
    else:
        net.apply(models.weights_init_uniform_rule)
        
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, eps=1e-08, weight_decay=0.0000005)

    X, te, metadata, nnz_tr, nnz_te = middlebury.load('imperfect', 'gray')
    nnz = nnz_tr
    print('Carregou')
    input()
    
    x_batch_p_tr = torch.FloatTensor((batch_size, channel_number, patch_height, patch_width)).to(device)
    x_batch_n_tr = torch.FloatTensor((batch_size, channel_number, patch_height, patch_width)).to(device)
    x_batch_p_tr_ = torch.FloatTensor(x_batch_p_tr.size())
    x_batch_n_tr_ = torch.FloatTensor(x_batch_n_tr.size())
    
    y_batch_p_tr = torch.FloatTensor(batch_size).to(device)
    y_batch_n_tr = torch.FloatTensor(batch_size).to(device)
    y_batch_p_tr_ = torch.FloatTensor(y_batch_p_tr.size())
    y_batch_n_tr_ = torch.FloatTensor(y_batch_n_tr.size())

    perm = torch.randperm(nnz.size()[0])

    time_start = time.time()

    for epoch in range(0, epochs_number):
        net.train()

        err_tr = 0
        err_tr_cnt = 0

        for t in range(0, nnz.size()[0] - int(BATCH_SIZE/2), int(BATCH_SIZE/2)):
            for i in range(0, int(BATCH_SIZE/2)):
                d_pos = random.uniform(-dataset_pos, dataset_pos)
                d_neg = random.uniform(dataset_neg_low, dataset_neg_high)

                if random.uniform() < 0.5:
                    d_neg = -d_neg
                
                # Augumentation
                s = 1
                scale = [1,1]
                hshear = 0
                trans = [0,0]
                phi = random.uniform(-ROTATE * math.pi / 180, ROTATE * math.pi / 180)
                brightness = 0
                contrast = random.uniform(1 / CONTRAST, CONTRAST)
                scale_ = [1,1]
                hshear_ = 0
                trans_ = [0,0]
                phi_ = phi + random.uniform(-D_ROTATE * math.pi / 180, D_ROTATE * math.pi / 180)
                brightness_ = 0
                contrast_ = contrast * random.uniform(1 / D_CONTRAST, D_CONTRAST)

                ind = perm[t + i]
                img = int(nnz[ind, 0].item())
                dim3 = int(nnz[ind, 1].item())
                dim4 = int(nnz[ind, 2].item())
                d = nnz[ind, 3].item()

                lenImg = len(X[img])    # Light qty

                if lenImg:
                    light = (random.randint(0,10000) % lenImg)

                    lenExp = X[img][light].shape[0]    # Exp qty
                    exp = (random.randint(1,10000) % lenExp)

                    aux = X[img][light][exp].shape[0]

                    light_ = light
                    exp_ = exp

                    if aux > 1:    
                        if random() < 0.2:
                            exp_ = (random.randint(1,10000) % lenExp)
                        
                        if random() < 0.2:
                            light_ = max(1, (random.randint(1,10000) % lenImg))

                        x0 = X[img][light][exp,0]
                        x1 = X[img][light_][exp_,1]

                        pair1Temp_d = utils.make_patch(x0, (patch_height, patch_width), dim4, dim3, device, scale, phi, trans, hshear, brightness, contrast, channel_size=channel_number)
                        pair2Temp_d = utils.make_patch(x1, (patch_height, patch_width), dim4 - d + d_pos, dim3, device, scale_, phi_, trans_, hshear_, brightness_, contrast_, channel_size=channel_number)
                        pair2TempN_d = utils.make_patch(x1, (patch_height, patch_width), dim4 - d + d_neg, dim3, device, scale_, phi_, trans_, hshear_, brightness_, contrast_, channel_size=channel_number)

                        x_batch_p_tr_[i * 2] = pair1Temp_d
                        x_batch_p_tr_[i * 2 + 1] = pair2Temp_d
                        
                        x_batch_n_tr_[i * 2] = pair1Temp_d
                        x_batch_n_tr_[i * 2 + 1] = pair2TempN_d

                        y_batch_p_tr_[i * 2] = 1
                        y_batch_n_tr_[i * 2] = 0

            x_batch_p_tr = x_batch_p_tr_.clone().detach()
            x_batch_n_tr = x_batch_n_tr_.clone().detach()
            y_batch_p_tr = y_batch_p_tr_.clone().detach()
            y_batch_n_tr = y_batch_n_tr_.clone().detach()

            optimizer.zero_grad()

            output = net(x_batch_p_tr, x_batch_n_tr)
            output = output.squeeze()

            output_s = output[::2]
            output_n = output[1::2]

            loss_p = criterion(output_s, y_batch_p_tr)
            loss_n = criterion(output_n, y_batch_n_tr)

            loss = loss_p + loss_n

            loss.backward()
            optimizer.step()

            err_tr += loss.item()
            err_tr_cnt += 1

            torch.save(net.state_dict(), weight_path)

        print('epoch\t%d loss:\t%.23f time lapsed:\t%.2f s' % (epoch, (err_tr/err_tr_cnt), time.time() - time_start))

if __name__ == "__main__":
    print("Training CNN...")

    train(
        batch_size = BATCH_SIZE,
        epochs_number = EPOCHS_NUMBER,
        device = DEVICE,
        dataset_neg_low = 1.5,
        dataset_neg_high = 6,
        dataset_pos = 0.5,
        patch_height = PATCH_HEIGHT,
        patch_width = PATCH_WIDTH,
        channel_number = CHANNEL_NUMBER
    )