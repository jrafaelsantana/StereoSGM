import os
import cv2
import numpy as np
import torch
import random
from utils import load_pfm
import math
import config
from pathlib import Path
from utils import save_obj, blur_image, census_transformation

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
CENSUS_KERNEL = int(settings.kernel_size_census)
BLUR_SIZE = int(settings.blur_size)


def generate_patches_training(
    directory_train,
    center_height,
    center_width,
    qt_correct,
    census_kernel,
    blur_size,
    dataset_neg_high=20.0,
    dataset_pos=0.5
):

    p = Path('.' + directory_train)
    subdirectories = [x for x in p.iterdir() if x.is_dir()]

    points = []
    pairs_list = []
    index_pair = 0

    for directory in subdirectories:
        print('Processing ' + directory.name)
        total = 0

        left = cv2.imread(directory._str + '/im0.png', 0)
        right = cv2.imread(directory._str + '/im1.png', 0)

        img_left = left.astype(np.float32)
        img_right = right.astype(np.float32)

        pfm_data = load_pfm(directory._str + '/disp0GT.pfm')
        mask = cv2.imread(directory._str + '/mask0nocc.png', 0)

        total_pixel = img_left.shape[0] * img_left.shape[1]
        sample = random.sample(range(0, total_pixel), total_pixel)

        sample_index = 0

        while (total < qt_correct and sample_index < total_pixel):
            i = int(sample[sample_index]/img_left.shape[1])
            j = int(sample[sample_index] % img_left.shape[1])

            pair1Temp = img_left[i-center_height:i +
                                 center_height+1, j-center_height:j+center_height+1]
            pair1TempValid = pair1Temp[pair1Temp == 255]
            pair1TempVar = np.std(pair1Temp)

            if(pair1TempVar > 30):
                if(i > center_height and i < img_left.shape[0] - center_height and j > (center_height) and j < img_left.shape[1] - center_width):
                    d = pfm_data[i, j]
                    valid = mask[i, j]

                    if not np.isinf(d) and valid != 255:
                        d_r = int(np.round(d))
                        if((j - d_r - dataset_pos - center_width) >= 0 and (j - d_r - dataset_neg_high - center_width) >= 0 and (j - d_r + dataset_pos + center_width) < img_left.shape[1] and (j - d_r + dataset_neg_high + center_width) < img_left.shape[1]):
                            dataset_neg_high = int(dataset_neg_high)

                            desvio_alto = True
                            d_I = -dataset_neg_high

                            while(desvio_alto and d_I < dataset_neg_high):
                                pair2Temp = img_right[i-center_height:i+center_height +
                                                      1, j-center_height-d_r+d_I:j+center_height+1-d_r+d_I]
                                d_I = d_I + 1
                                pair2TempValid = pair2Temp[pair2Temp == 255]
                                pair2TempVar = np.std(pair2Temp)

                                if(pair2TempVar <= 30):
                                    desvio_alto = False

                            if(desvio_alto == True):
                                points.append((i, j, d, index_pair))
                                total = total + 1

            sample_index = sample_index + 1

        index_pair = index_pair + 1
        pairs_list.append((img_left, img_right))

    return points, pairs_list


if __name__ == "__main__":
    dataset_dict = {}

    points, pair_list = generate_patches_training(
        settings.dataset_train,
        CENTER_PATCH_HEIGHT,
        CENTER_PATCH_WIDTH,
        QTY_CORRECT_TRAIN,
        CENSUS_KERNEL,
        BLUR_SIZE
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
