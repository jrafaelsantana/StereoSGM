import os
import cv2
import numpy as np
import torch
import random
from utils import load_pfm, write_pfm, save_obj
import math
import config
from pathlib import Path

if not os.path.exists('./obj'):
    os.mkdir('./obj')

settings = config.Config()

torch.manual_seed(int(settings.seed))

PATCH_WIDTH = int(settings.patch_width)
PATCH_HEIGHT = int(settings.patch_height)
DH = int(settings.height_stride)
DW = int(settings.width_stride)
CENTER_PATCH_WIDTH = int(PATCH_WIDTH/2)
CENTER_PATCH_HEIGHT = int(PATCH_HEIGHT/2)
QTY_CORRECT_TRAIN = int(settings.train_correct)
QTY_INCORRECT_TRAIN = int(settings.train_incorrect)
CHANNEL_NUMBER = int(settings.channel_number)


def generate_patches_training(
    directory_train,
    center_height,
    center_width,
    qt_correct,
    channel_number=1,
    dataset_neg_high=25.0,
    dataset_pos=0.5
):

    p = Path('.' + directory_train)
    subdirectories = [x for x in p.iterdir() if x.is_dir()]

    points = []
    pairs_list = []
    index_pair = 0

    for index, directory in enumerate(subdirectories):
        print('Processing ' + directory.name)
        total = 0

        if channel_number == 3:
            left = cv2.imread(directory._str + '/im0.png', cv2.COLOR_BGR2RGB)
            right = cv2.imread(directory._str + '/im1.png', cv2.COLOR_BGR2RGB)
        else:
            left = cv2.imread(directory._str + '/im0.png', 0)
            right = cv2.imread(directory._str + '/im1.png', 0)

        left = (left - left.mean()) / left.std()
        right = (right - right.mean()) / right.std()

        img_left = left.astype(np.float32)
        img_right = right.astype(np.float32)

        # mean_left = np.mean(img_left)
        # var_left = np.std(img_left)
        # img_left_d = (img_left - mean_left)/var_left

        # mean_right = np.mean(img_right)
        # var_right = np.std(img_right)
        # img_right_d = (img_right - mean_right)/var_right

        pfm_data = load_pfm(directory._str + '/disp0GT.pfm')
        # teste = pfm_data.astype('float32')

        # #cv2.imwrite('testevcs.pfm', teste, cv2.CV_32F)
        # write_pfm('teste.pfm', teste)
        # print('aqui')
        # input()
        
        mask = cv2.imread(directory._str + '/mask0nocc.png', 0)

        total_pixel = img_left.shape[0] * img_left.shape[1]
        sample = random.sample(range(0, total_pixel), total_pixel)

        sample_index = 0

        while (total < qt_correct and sample_index < total_pixel):
            i = int(sample[sample_index] / img_left.shape[1])
            j = int(sample[sample_index] % img_left.shape[1])

            start_x = j-center_height

            if start_x >=0: 
                if(i > center_height and i < img_left.shape[0] - center_height and j > (center_height) and j < img_left.shape[1] - center_width):
                    d = pfm_data[i, j]
                    valid = mask[i, j]

                    if not np.isinf(d) and valid != 255:
                        d_r = int(np.round(d))
                        if((j - d_r - dataset_pos - center_width) >= 0 and (j - d_r - dataset_neg_high - center_width) >= 0 and (j - d_r + dataset_pos + center_width) < img_left.shape[1] and (j - d_r + dataset_neg_high + center_width) < img_left.shape[1]):
                            dataset_neg_high = int(dataset_neg_high)

                            #desvio_alto = True
                            d_I = -dataset_neg_high

                            while(d_I < dataset_neg_high):
                                d_I = d_I + 1

                            #while(desvio_alto and d_I < dataset_neg_high):
                                #pair2Temp = img_right[i-center_height:i+center_height+1,j-center_height-d_r+d_I:j+center_height+1-d_r+d_I]
                                #d_I = d_I + 1

                                # pair2TempVar = np.std(pair2Temp) 
                                
                                # if(pair2TempVar <= 0.40): 
                                #     desvio_alto = False

                            # if(desvio_alto == True):
                            #     points.append((i,j,d,index_pair))
                            #     total = total + 1
                            points.append((i,j,d,index_pair))
                            total = total + 1

                sample_index = sample_index + 1
            else:
                sample_index = sample_index + 1

        index_pair = index_pair + 1
        #pairs_list.append((img_left_d, img_right_d))

        pairs_list.append((img_left, img_right))

    return points, pairs_list


if __name__ == "__main__":
    dataset_dict = {}

    points, pair_list = generate_patches_training(
        settings.dataset_train,
        CENTER_PATCH_HEIGHT,
        CENTER_PATCH_WIDTH,
        QTY_CORRECT_TRAIN,
        CHANNEL_NUMBER
    )

    points = torch.FloatTensor(points)
    limitP = math.ceil(points.shape[0]*0.90)

    point_split = torch.split(points, limitP)

    train_point = point_split[0]
    valid_point = point_split[1]

    dataset_dict['pair_list'] = pair_list
    dataset_dict['points_train'] = train_point
    dataset_dict['points_valid'] = valid_point

    save_obj(dataset_dict, 'database')
