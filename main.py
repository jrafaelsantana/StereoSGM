from paths import Paths
import utils
import config
import utils
import evaluate
import models

import numpy as np
from numba import jit
from pathlib import Path
import torch
import os
import cv2
import datetime

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
PENALTY_EQUAL_1 = float(settings.penalty_equal_1)
PENALTY_BIGGER_THEN_1 = float(settings.penalty_bigger_than_1)

PFM_DIR = '/home/rafael/Desenvolvimento/MiddleburySDK/MiddEval3/trainingQ/'

weight_path = 'weights/trainedweight.pth'


"""
    Semi-global matching

    Steps:
        1- Compute costs (Census transformation and Hamming distance)
        2- Compute left and right aggregation volume
        3- Select best disparity
        4- Evaluate
"""

net = models.Siamese(CHANNEL_NUMBER,1).to(DEVICE)

if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
else:
    print('Weights not found')


def compute_costs(left, right, max_disparity, patch_height, patch_width, channel_number, device):
    """
    Matching cost based on census transform and hamming distance.

    :param left: Left image.
    :param right: Right image.
    :param max_disparity: Maximum disparity.

    :return: An array [H, W, D] with the matching costs (Height, width and disparity).
    """

    print('Computing costs...')

    net.eval()

    height = left.shape[0]
    width = left.shape[1]
    c = int(patch_height/2)

    if (channel_number == 1):
        left = np.expand_dims(left, axis=2)
        right = np.expand_dims(right, axis=2)

    left = left.transpose((2, 0, 1))
    right = right.transpose((2, 0, 1))

    left = torch.FloatTensor(left).to(device)
    right = torch.FloatTensor(right).to(device)

    left = left.unsqueeze(0)
    right = right.unsqueeze(0)

    with torch.no_grad():
        begin_time = datetime.datetime.now()
        out1_small, out2_small, out1, out2 = net(left, right, training=False)
        print("Run CNN: {}".format(datetime.datetime.now() - begin_time))

        begin_time = datetime.datetime.now()
        
        # out1 = out1.squeeze().cpu().numpy()
        # out2 = out2.squeeze().cpu().numpy()
        # out1_small = out1_small.squeeze().cpu().numpy()
        # out2_small = out2_small.squeeze().cpu().numpy()

        out1 = out1.squeeze()
        out2 = out2.squeeze()
        out1_small = out1_small.squeeze()
        out2_small = out2_small.squeeze()

        costs = calc_costs(out1, out2, out1_small, out2_small)
        costs = costs.cpu().numpy()
        
        print("Costs: {}".format(datetime.datetime.now() - begin_time))

        return costs

'''@jit(nopython=True)
def calc_costs(out1, out2, out1_small, out2_small):
    max_disparity, height, width = out1.shape
    costs = np.zeros((height, width, max_disparity), dtype=np.float32)

    for y in range(0, height - 1):
        #print('Y ' + str(y))

        for x in range(0, width - 1):
            point_l = out1[:, y, x]
            point_l_small = out1_small[:, y, x]

            for nd in range(0, max_disparity):
                point_r = out2[:, y, x-nd]
                point_r_small = out2_small[:, y, x-nd]
                #result = np.sum((point_l - point_r) * (point_l - point_r))
                #result = np.abs(np.sum(point_l * point_r))
                #result = -np.sqrt(np.sum(np.power(point_l * point_r, 2)))
                #calc = point_l * point_r * point_l_small * point_r_small
                #result = np.sum(calc)

                #calc1 = point_l_small + point_r_small
                #calc2 = point_l + point_r

                #result = np.sqrt(np.sum((calc1 - calc2) * (calc1 - calc2)))
                result = np.sqrt(np.sum((point_l_small - point_r_small) * (point_l_small - point_r_small)))
                #result = np.abs(np.sum(calc1 + calc2))
                #result = np.abs(np.sum(calc1 - calc2))

                #result = np.sqrt(np.sum((point_l - point_r) * (point_l - point_r)))
                #result = np.sqrt(np.sum((point_l_small - point_r_small) * (point_l_small - point_r_small)))

                #calc1 = point_l @ point_r
                #calc2 = point_l_small @ point_r_small
                #calc = calc1 + calc2

                # print(calc)
                # print(calc1)
                # print(calc2)
                # print()
                # input()

                #result = np.sum((point_l * point_r) + (point_l_small * point_r_small))
                #print(result)
                #input()
                    
                costs[y, x, nd] = result

    return costs
'''
def calc_costs(out1, out2, out1_small, out2_small):
    max_disparity, height, width = out1.shape
    costs = torch.zeros((height, width, max_disparity), dtype=torch.float32).to(DEVICE)

    for nd in range(0, int(max_disparity)):
        for x in range(0, int(width) - 1)[::-1]:
            point_l = out1[:, :, x] # [Features, Y]
            point_l_small = out1_small[:, :, x] # [Features, Y]

            point_r = out2[:, :, x-nd] # [Features, Y]
            point_r_small = out2_small[:, :, x-nd] # [Features, Y]
            
            #result = torch.sqrt(torch.sum((point_l_small - point_r_small) * (point_l_small - point_r_small), 0))
            
            #calc1 = point_l_small - point_r_small
            #calc2 = point_l - point_r

            #result = np.sum((calc1 - calc2) * (calc1 - calc2), 0)
            
            # print(point_l.shape)
            # print(point_r.shape)
            # print(point_l_small.shape)
            # print(point_r_small.shape)

            #conc_mat = torch.cat((point_l, point_l_small, point_r, point_r_small), 0) # [Features * 4, Y]
            conc_mat = torch.cat((point_l_small, point_r_small), 0) 
            #conc_mat = torch.abs(point_l_small - point_r_small)

            # print(conc_mat.shape)
            conc_mat = conc_mat.transpose(1,0) # [Y, Features * 4] torch.Size([279, 512])
            # print(conc_mat.shape)
            result = net.linear(conc_mat) # [Y, 1]
            result = result.squeeze() # [Y]
            #result = torch.pow(result, 2)

            costs[:, x, nd] = result
    return costs

    # for y in range(0, int(height) - 1):
    #     print(y)
    #     for x in range(0, int(width) - 1):
    #         point_l = out1[:, y, x]
    #         point_l_small = out1_small[:, y, x]

    #         for nd in range(0, int(max_disparity)):
    #             point_r = out2[:, y, x-nd]
    #             point_r_small = out2_small[:, y, x-nd]

    #             conc_mat = torch.cat((point_l, point_r, point_l_small, point_r_small))
    #             result = model.linear(conc_mat)
                        
    #             costs[y, x, nd] = result

    # return costs


def compute_aggregation(cost, paths, p1, p2):
    """
    Aggregates matching costs for N possible directions.

    :param cost: array containing the matching costs.
    :param paths: structure containing all directions in which to aggregate costs.
    :param p1: Penalty equals 1.
    :param p2: Penalty bigger then 1.

    :return: H x W x D x N array of matching cost for all defined directions.
    """

    print('Computing aggregation costs...')
    begin_time = datetime.datetime.now()

    height = cost.shape[0]
    width = cost.shape[1]
    disparities = cost.shape[2]

    start = -(height - 1)
    end = width - 1

    aggregation_volume = np.zeros(
        shape=(height, width, disparities, paths.size), dtype=cost.dtype)

    path_id = 0
    for path in paths.effective_paths:

        main_aggregation = np.zeros(
            shape=(height, width, disparities), dtype=cost.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == paths.S.direction:
            for x in range(0, width):
                south = cost[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = utils.get_path_cost(
                    south, 1, p1, p2)
                opposite_aggregation[:, x, :] = np.flip(
                    utils.get_path_cost(north, 1, p1, p2), axis=0)

        if main.direction == paths.E.direction:
            for y in range(0, height):
                east = cost[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = utils.get_path_cost(
                    east, 1, p1, p2)
                opposite_aggregation[y, :, :] = np.flip(
                    utils.get_path_cost(west, 1, p1, p2), axis=0)

        if main.direction == paths.SE.direction:
            for offset in range(start, end):
                south_east = cost.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = utils.get_indices(
                    paths, offset, dim, paths.SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = utils.get_path_cost(
                    south_east, 1, p1, p2)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = utils.get_path_cost(
                    north_west, 1, p1, p2)

        if main.direction == paths.SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = utils.get_indices(
                    paths, offset, dim, paths.SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = utils.get_path_cost(
                    south_west, 1, p1, p2)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = utils.get_path_cost(
                    north_east, 1, p1, p2)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2
    print("Compute aggregation: {}".format(datetime.datetime.now() - begin_time))

    return aggregation_volume


def select_best_disparity(aggregation_cost, max_disparity):
    """
    Return the best disparity.

    :param aggregation_cost: H x W x D x N array of matching cost for all defined directions.

    :return: disparity image.
    """

    print('Selecting best disparity...')
    begin_time = datetime.datetime.now()

    volume = np.sum(aggregation_cost, axis=3)
    disparity_map = np.argmin(volume, axis=2)

    print("Best disparity: {}".format(datetime.datetime.now() - begin_time))

    return disparity_map


def sgm(directory):
    paths = Paths()

    height, width, max_disparity, cam1, cam2 = utils.parseCalib(directory._str + '/calib.txt')

    gt_file = utils.load_pfm(directory._str + '/disp0GT.pfm')

    if CHANNEL_NUMBER == 3:
        left = cv2.imread(directory._str + '/im0.png', cv2.COLOR_BGR2RGB)
        right = cv2.imread(directory._str + '/im1.png', cv2.COLOR_BGR2RGB)
    else:
        left = cv2.imread(directory._str + '/im0.png', 0)
        right = cv2.imread(directory._str + '/im1.png', 0)

    left = (left - left.mean()) / left.std()
    right = (right - right.mean()) / right.std()

    costs = compute_costs(left, right, max_disparity,
                          PATCH_HEIGHT, PATCH_WIDTH, CHANNEL_NUMBER, DEVICE)

    aggregation = compute_aggregation(
        costs, paths, PENALTY_EQUAL_1, PENALTY_BIGGER_THEN_1)

    best_disp = select_best_disparity(aggregation, max_disparity)
    pfm = best_disp.astype(np.float32)
    utils.write_pfm(PFM_DIR + directory.name + '/disp0DUPLAJANELA7.pfm', pfm)
    
    utils.saveDisparity(np.uint8(best_disp), 'resultados/' + directory.name + '_7.png')

    # print("Evaluate")
    # recall = evaluate.recall(best_disp, gt_file, max_disparity)
    # print('\tRecall = {:.2f}%'.format(recall * 100.0))


if __name__ == "__main__":
    p = Path('.' + settings.dataset_train)
    subdirectories = [x for x in p.iterdir() if x.is_dir()]

    # for name, param in net.named_parameters():
    #     if 'full' in name:
    #         print(name)
    #         print(param)
    #         print()
    
    # input()

    for directory in subdirectories:
        if directory.name == 'Adirondack':  # For tests only
            sgm(directory)
