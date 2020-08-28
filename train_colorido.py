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

if not os.path.exists('weights/'):
    os.mkdir('weights/')

weight_path = 'weights/trainedweight.pth'


def train(batch_size, epochs_number, pair_list, points_train, points_valid, device, weight_path=None, dataset_neg_low=2.5, dataset_neg_high=6.0, dataset_pos=0.5, center_height=5, center_width=5, patch_height=11, patch_width=11, channel_number=1, iter_no_impro=100):
    net = models.Siamese(channel_number).to(device)
    loss_fn = torch.nn.MarginRankingLoss(0.2)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0000001, eps=1e-08, weight_decay=0.0000005)

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

    pair_list_temp = np.zeros((len(pair_list), 2, channel_number, max_rows, max_cols))

    for index, pair in enumerate(pair_list):
        height = max([len(rows) for rows in pair])
        width = max([len(cols) for rows in pair for cols in rows])

        pair_temp0 = np.asarray(pair[0])
        pair_temp0 = pair_temp0.transpose((2, 0, 1))

        pair_temp1 = np.asarray(pair[1])
        pair_temp1 = pair_temp1.transpose((2, 0, 1))

        pair_list_temp[index, 0, :, 0:height, 0:width] = pair_temp0
        pair_list_temp[index, 1, :, 0:height, 0:width] = pair_temp1

    pair_list = torch.tensor(pair_list_temp).to(device)
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

                images1_batch = torch.FloatTensor(
                    2*batch_temp, channel_number, patch_width, patch_height).to(device)
                images2_batch = torch.FloatTensor(
                    2*batch_temp, channel_number, patch_width, patch_height).to(device)
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

                    if channel_number == 3:
                        pair1Temp_d = images1[:,i-center_height:i+center_height+1,j-center_height:j+center_height+1]
                        pair2Temp_d = images2[:,i-center_height:i+center_height+1,j + pos_d - center_height:j + pos_d + center_height + 1]
                        pair2TempN_d = images2[:,i-center_height:i+center_height+1,j + neg_d - center_height: j + neg_d + center_height + 1 ]

                        '''pair2Temp_d_p_aug = np.uint8(pair2Temp_d.cpu())
                        pair2Temp_d_p_aug = pair2Temp_d_p_aug.transpose((2, 1, 0))

                        if random.uniform(0, 1) < 0.3:
                            if random.uniform(0, 1) < 0.5:
                                pair2Temp_d_p_aug = utils.flip(pair2Temp_d_p_aug, True)
                            else:
                                pair2Temp_d_p_aug = utils.rotate(pair2Temp_d_p_aug, True)

                        pair2Temp_d_p_aug = pair2Temp_d_p_aug.transpose((2, 0, 1))
                        pair2Temp_d = torch.tensor(pair2Temp_d_p_aug, dtype=torch.float64).to(device)'''

                    else:
                        pair1Temp_d = images1[i-center_height:i+center_height+1,j-center_height:j+center_height+1]
                        pair2Temp_d = images2[i-center_height:i+center_height+1,j-center_height+pos_d:j+center_height+pos_d+1]
                        pair2TempN_d = images2[i-center_height:i+center_height+1,j-center_height+neg_d:j+center_height+neg_d+1]
                            
                    images1_batch[2*patch_id+patch_order] = pair1Temp_d
                    images2_batch[2*patch_id+patch_order] = pair2Temp_d
                    images1_batch[2*patch_id+1-patch_order] = pair1Temp_d
                    images2_batch[2*patch_id+1-patch_order] = pair2TempN_d

                    target[patch_id] = -1.0 + 2*patch_order

                #print(images1_batch.shape)
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

                    images1_batch = torch.FloatTensor(
                        2*batch_temp, channel_number, patch_width, patch_height).to(device)
                    images2_batch = torch.FloatTensor(
                        2*batch_temp, channel_number, patch_width, patch_height).to(device)
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

                        if channel_number == 3:
                            pair1Temp_d = images1[:,i-center_height:i+center_height+1,j-center_height:j+center_height+1]
                            pair2Temp_d = images2[:,i-center_height:i+center_height+1,j + pos_d - center_height:j + pos_d + center_height + 1]
                            pair2TempN_d = images2[:,i-center_height:i+center_height+1,j + neg_d - center_height: j + neg_d + center_height + 1 ]
                        else:
                            pair1Temp_d = images1[i-center_height:i+center_height+1,j-center_height:j+center_height+1]
                            pair2Temp_d = images2[i-center_height:i+center_height+1,j + pos_d - center_height:j + pos_d + center_height + 1]
                            pair2TempN_d = images2[i-center_height:i+center_height+1,j + neg_d - center_height: j + neg_d + center_height + 1 ]
                            
                        images1_batch[2*patch_id+patch_order] = pair1Temp_d
                        images2_batch[2*patch_id+patch_order] = pair2Temp_d
                        images1_batch[2*patch_id+1-patch_order] = pair1Temp_d
                        images2_batch[2*patch_id+1-patch_order] = pair2TempN_d

                        target[patch_id] = -1.0 + 2*patch_order

                    output_v = net(images1_batch, images2_batch)

                    output_s_v = output_v[::2]
                    output_n_v = output_v[1::2]

                    loss = loss_fn(output_s_v, output_n_v, target)

                    loss_val += loss.item()
                    loss_val_cnt += 1

                    diff = ((output_n_v - output_s_v).view(-1)
                            * target).view(-1, 1)

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
        BATCH_SIZE,
        EPOCHS_NUMBER,
        pair_list,
        points_train,
        points_valid,
        DEVICE,
        weight_path,
        1,
        6,
        0.5,
        CENTER_PATCH_HEIGHT,
        CENTER_PATCH_WIDTH,
        PATCH_HEIGHT,
        PATCH_WIDTH,
        CHANNEL_NUMBER
    )
