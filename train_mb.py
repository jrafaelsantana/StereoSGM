import random

from torch._C import dtype
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

    criterion = torch.nn.MarginRankingLoss(0.5).to(device)

    if(weight_path != None and os.path.exists(weight_path)):
        net.load_state_dict(torch.load(weight_path))
    else:
        net.apply(models.weights_init_uniform_rule)
        
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, eps=1e-08, weight_decay=0.0000005)

    X, te, metadata, nnz_tr, nnz_te = middlebury.load('imperfect', 'gray', device)
    nnz = nnz_tr
    
    x_batch_p_tr = torch.zeros((batch_size*2, channel_number, patch_height, patch_width), dtype=torch.float32).to(device)
    x_batch_n_tr = torch.zeros((batch_size*2, channel_number, patch_height, patch_width), dtype=torch.float32).to(device)

    target = torch.linspace(1.0, 1.0, dtype=torch.float, steps=int(batch_size), device=device)

    time_start = time.time()

    counter_i = 0

    for epoch in range(0, epochs_number):
        net.train()

        err_tr = 0
        err_tr_cnt = 0

        perm = torch.randperm(nnz.size()[0])

        for t in range(0, 2048 - int(batch_size/2), int(batch_size/2)):
        #for t in range(0, nnz.size()[0] - int(batch_size/2), int(batch_size/2)):
            
            for i in range(0, int(batch_size/2)):
                d_pos = random.uniform(-dataset_pos, dataset_pos)
                d_neg = random.uniform(dataset_neg_low, dataset_neg_high)

                if random.random() < 0.5:
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

                light = (random.randint(0,10000) % max(2, len(X[img])) - 1) + 1
                exp = (random.randint(0,10000) % X[img][light].shape[0])

                light_ = light
                exp_ = exp

                if random.random() < 0.2:
                    exp_ = (random.randint(1,10000) % X[img][light].shape[0])
                
                if random.random() < 0.2:
                    light_ = max(1, light - 1)

                print(len(X[img]))
                print(X[img][light].shape)
                print(light)
                print(light_)
                print(exp)
                print(exp_)
                print(X[img][light][exp].shape)
                input()

                x0 = X[img][light][exp,0]
                x1 = X[img][light_][exp_,1]

                pair1Temp_d = utils.make_patch(x0, (patch_height, patch_width), dim4, dim3, device, scale, phi, trans, hshear, brightness, contrast, channel_size=channel_number)
                pair2Temp_d = utils.make_patch(x1, (patch_height, patch_width), dim4 - d + d_pos, dim3, device, scale_, phi_, trans_, hshear_, brightness_, contrast_, channel_size=channel_number)
                pair2TempN_d = utils.make_patch(x1, (patch_height, patch_width), dim4 - d + d_neg, dim3, device, scale_, phi_, trans_, hshear_, brightness_, contrast_, channel_size=channel_number)

                x_batch_p_tr[counter_i * 2] = pair1Temp_d
                x_batch_p_tr[counter_i * 2 + 1] = pair2Temp_d
                
                x_batch_n_tr[counter_i * 2] = pair1Temp_d
                x_batch_n_tr[counter_i * 2 + 1] = pair2TempN_d

                target[counter_i] = -1.0

                counter_i = counter_i + 1

                if(counter_i == 127):        

                    optimizer.zero_grad()

                    output = net(x_batch_p_tr, x_batch_n_tr)
                    output = output.squeeze()

                    output_s = output[::2]
                    output_n = output[1::2]

                    loss = criterion(output_s, output_n, target)

                    loss.backward()
                    optimizer.step()

                    err_tr += loss.item()
                    err_tr_cnt += 1

                    counter_i = 0

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