import models
import torch
import time
import numpy as np
import cv2
import os
import preprocess
import utils
import random
import config
import models
import math

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
HFLIP = float(settings.HFLIP)
VFLIP = float(settings.VFLIP)
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


def train(batch_size, epochs_number, pair_list, points_train, points_valid, device, weight_path=None, dataset_neg_low=2.5, dataset_neg_high=6.0, dataset_pos=0.5, center_height=5, center_width=5, patch_height=11, patch_width=11, channel_number=1, iter_no_impro=100):
    assert(HSCALE <= 1 and SCALE <= 1)
    assert(CONTRAST >= 1 and D_CONTRAST >= 1)
    
    net = models.Siamese(channel_number).to(device)
    loss_fn = torch.nn.MarginRankingLoss(0.2)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-08, weight_decay=0.0000005)

    if(weight_path != None and os.path.exists(weight_path)):
        net.load_state_dict(torch.load(weight_path))
    else:
        net.apply(models.weights_init_uniform_rule)

    time_start = time.time()
    iter_no_impro_counter = 0
    smaller_error = 1000000
    loss_val = 0

    max_rows = max([len(rows) for image in pair_list for rows in image])
    max_cols = max([len(cols) for image in pair_list for rows in image for cols in rows])

    pair_list_temp = np.zeros((len(pair_list), 2, max_rows, max_cols, channel_number))

    for index, pair in enumerate(pair_list):
        height = max([len(rows) for rows in pair])
        width = max([len(cols) for rows in pair for cols in rows])

        pair_temp0 = np.asarray(pair[0])
        pair_temp1 = np.asarray(pair[1])

        if (channel_number == 1):
            pair_temp0 = np.expand_dims(pair_temp0, axis=2)
            pair_temp1 = np.expand_dims(pair_temp1, axis=2)

        # pair_temp0 = pair_temp0.transpose((2, 0, 1))
        # pair_temp1 = pair_temp1.transpose((2, 0, 1))

        pair_list_temp[index, 0, 0:height, 0:width, :] = pair_temp0
        pair_list_temp[index, 1, 0:height, 0:width, :] = pair_temp1

    pair_list = pair_list_temp
    points_train = points_train.clone().detach().to(device)
    points_valid = points_valid.clone().detach().to(device)

    points_split = torch.split(points_train, batch_size, dim=0)
    points_valid_split = torch.split(points_valid, batch_size, dim=0)

    begin_val = 1

    # TRAIN
    for epoch in range(0, epochs_number):

        if(begin_val == 0):
            net.train()

            loss_tr_cnt = 0
            loss_tr = 0
            total_batches = len(points_split)
            sample = random.sample(range(0, total_batches), total_batches)

            for batch_id in sample:
                batch_temp = len(points_split[batch_id])

                images1_batch = np.zeros((2*batch_temp, patch_width, patch_height, channel_number))
                images2_batch = np.zeros((2*batch_temp, patch_width, patch_height, channel_number))
                target = np.linspace(1.0, 1.0, num=batch_temp)
                target = torch.from_numpy(target).float().to(device)

                patch_sample = random.sample(range(0, batch_temp, 1), int(batch_temp))
                patch_order = 0

                for patch_id in range(0, batch_temp):
                    i, j, d, img_idx = points_split[batch_id][patch_sample[patch_id]]
                    img_idx = int(img_idx.item())

                    i = int(i.item())
                    j = int(j.item())
                    d = d.item()

                    pos_offset = random.uniform(-dataset_pos, dataset_pos)
                    pos_d = int(np.round(-d + pos_offset))
                    neg_offset = random.uniform(dataset_neg_low, dataset_neg_high)
                    
                    if random.uniform(0, 1) < 0.5:
                        neg_offset = -neg_offset

                    neg_d = int(np.round(-d + neg_offset))

                    images1 = pair_list[img_idx][0]
                    images2 = pair_list[img_idx][1]

                    # New augumentation
                    s = random.uniform(SCALE, 1)
                    scale = [s * random.uniform(HSCALE, 1), s]
                        
                    hshear = random.uniform(-HSHEAR, HSHEAR)
                    trans = [random.uniform(-TRANS, TRANS), random.uniform(-TRANS, TRANS)]
                    phi = random.uniform(-ROTATE * math.pi / 180, ROTATE * math.pi / 180)
                    brightness = random.uniform(-BRIGHTNESS, BRIGHTNESS)
                    
                    contrast = random.uniform(1 / CONTRAST, CONTRAST)

                    scale_ = [scale[0] * random.uniform(D_HSCALE, 1), scale[1]]
                    hshear_ = hshear + random.uniform(-D_HSHEAR, D_HSHEAR)
                    trans_ = [trans[0], trans[1] + random.uniform(-D_VTRANS, D_VTRANS)]
                    phi_ = phi + random.uniform(-D_ROTATE * math.pi / 180, D_ROTATE * math.pi / 180)
                    brightness_ = brightness + random.uniform(-D_BRIGHTNESS, D_BRIGHTNESS)
                    contrast_ = contrast * random.uniform(1 / D_CONTRAST, D_CONTRAST)

                    # print(j)
                    # print(d)
                    # print(pos_d)
                    # print(neg_d)
                    # print(j + pos_d)
                    # print(j + neg_d)
                    # print()

                    patch1 = utils.make_patch(images1, patch_height, i, j, scale, phi, trans, hshear, brightness, contrast)
                    patch2 = utils.make_patch(images2, patch_height, i, j + pos_d, scale_, phi_, trans_, hshear_, brightness_, contrast_)
                    patch3 = utils.make_patch(images2, patch_height, i, j + neg_d, scale_, phi_, trans_, hshear_, brightness_, contrast_)

                    pair1Temp_d = patch1[0:patch_height, 0:patch_height]
                    pair2Temp_d = patch2[0:patch_height, 0:patch_height]
                    pair2TempN_d = patch3[0:patch_height, 0:patch_height]

                    # cv2.imshow('referencia', np.uint8(pair1Temp_d))
                    # cv2.imshow('correta', np.uint8(pair2Temp_d))
                    # cv2.imshow('incorreta', np.uint8(pair2TempN_d))
                    # cv2.waitKey()

                    '''if channel_number == 3:
                        patch1 = utils.make_patch(images1, center_height, i, j, scale, phi, trans, hshear, brightness, contrast)
                        patch2 = utils.make_patch(images2, center_height, i, j - d + pos_d, scale_, phi_, trans_, hshear_, brightness_, contrast_)
                        patch3 = utils.make_patch(images2, center_height, i, j - d + neg_d, scale_, phi_, trans_, hshear_, brightness_, contrast_)

                        pair1Temp_d = patch1[:, 0:center_height*2, 0:center_height*2]
                        pair2Temp_d = patch2[:, 0:center_height*2, 0:center_height*2]
                        pair2TempN_d = patch3[:, 0:center_height*2, 0:center_height*2]

                        # Augumentation
                        if random.uniform(0, 1) < 0.3:
                            patches_sim += 1
                            pair2Temp_d_p_aug = np.uint8(pair2Temp_d.cpu())
                            pair2Temp_d_p_aug = pair2Temp_d_p_aug.transpose((2, 1, 0))

                            augumentation_option = random.uniform(0, 4)

                            if augumentation_option == 0:
                                pair2Temp_d_p_aug = utils.horizontal_shift(pair2Temp_d_p_aug, AUGUMENTATION_HSHIFT)
                            elif augumentation_option == 1:
                                pair2Temp_d_p_aug = utils.brightness(pair2Temp_d_p_aug, AUGUMENTATION_BRIGHT_LOW, AUGUMENTATION_BRIGHT_HIGH)
                            #elif augumentation_option == 2:
                                #pair2Temp_d_p_aug = utils.zoom(pair2Temp_d_p_aug, AUGUMENTATION_ZOOM)
                            elif augumentation_option == 3:
                                pair2Temp_d_p_aug = utils.channel_shift(pair2Temp_d_p_aug, AUGUMENTATION_CHNSHIFT)
                            elif augumentation_option == 4:
                                pair2Temp_d_p_aug = utils.rotation(pair2Temp_d_p_aug, AUGUMENTATION_ROTANGLE)

                            pair2Temp_d_p_aug = pair2Temp_d_p_aug.transpose((2, 0, 1))
                            pair2Temp_d = torch.tensor(pair2Temp_d_p_aug, dtype=torch.float64).to(device)
                        else:
                            patches_nao += 1

                    else:
                        pair1Temp_d = images1[i-center_height:i+center_height+1,j-center_height:j+center_height+1]
                        pair2Temp_d = images2[i-center_height:i+center_height+1,j-center_height+pos_d:j+center_height+pos_d+1]
                        pair2TempN_d = images2[i-center_height:i+center_height+1,j-center_height+neg_d:j+center_height+neg_d+1]'''

                    images1_batch[2*patch_id+patch_order] = pair1Temp_d
                    images2_batch[2*patch_id+patch_order] = pair2Temp_d
                    images1_batch[2*patch_id+1-patch_order] = pair1Temp_d
                    images2_batch[2*patch_id+1-patch_order] = pair2TempN_d

                    target[patch_id] = -1.0 + 2*patch_order

                #print(images1_batch.shape)
                images1_batch = images1_batch.transpose((0, 3, 1, 2))
                images2_batch = images2_batch.transpose((0, 3, 1, 2))

                images1_batch = torch.FloatTensor(images1_batch).to(device)
                images2_batch = torch.FloatTensor(images2_batch).to(device)

                output = net(images1_batch, images2_batch)

                output_s = output[::2]
                output_n = output[1::2]

                loss = loss_fn(output_s, output_n, target)

                loss.backward()
                optimizer.step()

                loss_tr += loss.item()
                loss_tr_cnt += 1

            avg_loss = loss_tr/loss_tr_cnt
        else:
            begin_val = 0 
            avg_loss = -1

        # TEST
        if((epoch+1) % 1 == 0):
            net.eval()

            err_tr_cnt = 0
            err_tr = 0
            loss_val_cnt = 0
            loss_val = 0

            with torch.no_grad():
                total_batches_valid = len(points_valid_split)
                sample_valid = random.sample(
                    range(0, total_batches_valid), total_batches_valid)

                for batch_id in sample_valid:
                    batch_temp = len(points_valid_split[batch_id])

                    images1_batch = np.zeros((2*batch_temp, patch_width, patch_height, channel_number))
                    images2_batch = np.zeros((2*batch_temp, patch_width, patch_height, channel_number))
                    target = np.linspace(1.0, 1.0, num=batch_temp)
                    target = torch.from_numpy(target).float().to(device)

                    patch_sample = random.sample(range(0, batch_temp, 1), int(batch_temp))
                    patch_order = 1

                    for patch_id in range(0, batch_temp):
                        i, j, d, img_idx = points_valid_split[batch_id][patch_sample[patch_id]]
                        img_idx = int(img_idx.item())

                        i = int(i.item())
                        j = int(j.item())
                        d = d.item()

                        pos_offset = 0
                        pos_d = int(np.round(-d + pos_offset))
                        neg_offset = random.uniform(1, dataset_neg_high)
                        
                        if random.uniform(0,1) < 0.5:
                           neg_offset = -neg_offset

                        neg_d = int(np.round(-d + neg_offset)) 

                        images1 = pair_list[img_idx][0]
                        images2 = pair_list[img_idx][1]

                        pair1Temp_d = images1[i-center_height:i+center_height+1,j-center_height:j+center_height+1]
                        pair2Temp_d = images2[i-center_height:i+center_height+1,j + pos_d - center_height:j + pos_d + center_height + 1]
                        pair2TempN_d = images2[i-center_height:i+center_height+1,j + neg_d - center_height:j + neg_d + center_height + 1 ]

                        pair1Temp_d = images1[i-center_height:i+center_height+1,j-center_height:j+center_height+1]
                        pair2Temp_d = images2[i-center_height:i+center_height+1,j + pos_d - center_height:j + pos_d + center_height + 1]
                        pair2TempN_d = images2[i-center_height:i+center_height+1,j + neg_d - center_height:j + neg_d + center_height + 1 ]

                        '''if channel_number == 3:
                            pair1Temp_d = images1[:,i-center_height:i+center_height+1,j-center_height:j+center_height+1]
                            pair2Temp_d = images2[:,i-center_height:i+center_height+1,j + pos_d - center_height:j + pos_d + center_height + 1]
                            pair2TempN_d = images2[:,i-center_height:i+center_height+1,j + neg_d - center_height: j + neg_d + center_height + 1 ]
                        else:
                            pair1Temp_d = images1[i-center_height:i+center_height+1,j-center_height:j+center_height+1]
                            pair2Temp_d = images2[i-center_height:i+center_height+1,j + pos_d - center_height:j + pos_d + center_height + 1]
                            pair2TempN_d = images2[i-center_height:i+center_height+1,j + neg_d - center_height: j + neg_d + center_height + 1 ]'''
                            
                        images1_batch[2*patch_id+patch_order] = pair1Temp_d
                        images2_batch[2*patch_id+patch_order] = pair2Temp_d
                        images1_batch[2*patch_id+1-patch_order] = pair1Temp_d
                        images2_batch[2*patch_id+1-patch_order] = pair2TempN_d

                        target[patch_id] = -1.0 + 2*patch_order
                    
                    images1_batch = images1_batch.transpose((0, 3, 1, 2))
                    images2_batch = images2_batch.transpose((0, 3, 1, 2))

                    images1_batch = torch.FloatTensor(images1_batch).to(device)
                    images2_batch = torch.FloatTensor(images2_batch).to(device)

                    output_v = net(images1_batch, images2_batch)

                    output_s_v = output_v[::2]
                    output_n_v = output_v[1::2]

                    loss = loss_fn(output_s_v, output_n_v, target)

                    loss_val += loss.item()
                    loss_val_cnt += 1

                    diff = ((output_n_v - output_s_v).view(-1)* target).view(-1, 1)

                    comp = torch.ge(diff, 0)

                    err_tr += comp.sum().item()
                    err_tr_cnt += comp.shape[0]

                avg_err = err_tr/err_tr_cnt

                avg_val_loss = loss_val/loss_val_cnt

                '''if(smaller_error > avg_val_loss):
                    smaller_error = avg_val_loss
                    iter_no_impro_counter = 0
                    torch.save(net.state_dict(), weight_path)
                else:
                    iter_no_impro_counter = iter_no_impro_counter + 1'''

                torch.save(net.state_dict(), weight_path)

            print('epoch\t%d loss:\t%.23f val_loss:\t%.5f error:\t%.5f time lapsed:\t%.2f s' % (
                epoch, avg_loss, avg_val_loss, avg_err, time.time() - time_start))
        else:
            print('epoch\t%d loss:\t%.23f time lapsed:\t%.2f s' %
                  (epoch, avg_loss, time.time() - time_start))


if __name__ == "__main__":
    dataset_dict = utils.load_obj('database')

    pair_list = dataset_dict['pair_list']
    points_train = dataset_dict['points_train']
    points_valid = dataset_dict['points_valid']

    print("Training CNN...")

    train(
        batch_size = BATCH_SIZE,
        epochs_number = EPOCHS_NUMBER,
        pair_list = pair_list,
        points_train = points_train,
        points_valid = points_valid,
        device = DEVICE,
        weight_path = weight_path,
        dataset_neg_low = 3,
        dataset_neg_high = 10,
        dataset_pos = 0.5,
        center_height = CENTER_PATCH_HEIGHT,
        center_width = CENTER_PATCH_WIDTH,
        patch_height = PATCH_HEIGHT,
        patch_width = PATCH_WIDTH,
        channel_number = CHANNEL_NUMBER
    )
