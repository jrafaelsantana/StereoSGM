import numpy as np

from paths import Paths
from utils import census_transformation, hamming_distance, normalize_image, get_path_cost, get_indices

from cv2 import imread, imshow, waitKey

"""
    Semi-global matching

    Steps:
        1- Compute costs (Census transformation and Hamming distance)
        2- Compute left and right aggregation volume
        3- Select best disparity
        4- Evaluate
"""


def compute_costs(left, right, kernel_size_census, max_disparity):
    """
    Matching cost based on census transform and hamming distance.

    :param left: Left image.
    :param right: Right image.
    :param kernel_size_census: Dictionary with height and width of census kernel size.
    :param max_disparity: Maximum disparity.

    :return: An array [H, W, D] with the matching costs (Height, width and disparity).
    """

    height = left.shape[0]
    width = left.shape[1]
    census_kernel_height = kernel_size_census[0]
    census_kernel_width = kernel_size_census[1]

    y_offset = int(census_kernel_height / 2)
    x_offset = int(census_kernel_width / 2)

    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    # Census transformation

    left_census_values = census_transformation(
        left, census_kernel_height, census_kernel_width)
    right_census_values = census_transformation(
        right, census_kernel_height, census_kernel_width)

    # Hamming distance

    left_cost = np.zeros(
        shape=(height, width, max_disparity))
    right_cost = np.zeros(
        shape=(height, width, max_disparity))

    left_cost, right_cost = hamming_distance(
        left_census_values, right_census_values, max_disparity, x_offset, y_offset)

    return left_cost, right_cost


def compute_aggregation(cost, paths, p1, p2):
    """
    Aggregates matching costs for N possible directions.

    :param cost: array containing the matching costs.
    :param paths: structure containing all directions in which to aggregate costs.
    :param p1: Penalty equals 1.
    :param p2: Penalty bigger then 1.

    :return: H x W x D x N array of matching cost for all defined directions.
    """
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
                main_aggregation[:, x, :] = get_path_cost(south, 1, p1, p2)
                opposite_aggregation[:, x, :] = np.flip(
                    get_path_cost(north, 1, p1, p2), axis=0)

        if main.direction == paths.E.direction:
            for y in range(0, height):
                east = cost[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1, p1, p2)
                opposite_aggregation[y, :, :] = np.flip(
                    get_path_cost(west, 1, p1, p2), axis=0)

        if main.direction == paths.SE.direction:
            for offset in range(start, end):
                south_east = cost.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(
                    paths, offset, dim, paths.SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(
                    south_east, 1, p1, p2)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(
                    north_west, 1, p1, p2)

        if main.direction == paths.SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(
                    paths, offset, dim, paths.SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(
                    south_west, 1, p1, p2)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(
                    north_east, 1, p1, p2)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

    return aggregation_volume


def select_best_disparity(aggregation_cost, max_disparity):
    """
    Return the best disparity.

    :param aggregation_cost: H x W x D x N array of matching cost for all defined directions.

    :return: disparity image.
    """

    volume = np.sum(aggregation_cost, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return np.uint8(normalize_image(disparity_map, max_disparity))


def sgm():
    paths = Paths()

    left = imread('im0.png', 0)
    right = imread('im1.png', 0)

    print('Aqui')
    left_cost, right_cost = compute_costs(left, right, (5, 5), 64)

    left_aggregation = compute_aggregation(
        left_cost, paths, 10, 120)
    right_aggregation = compute_aggregation(
        right_cost, paths, 10, 120)

    left_disparity = select_best_disparity(left_aggregation, 64)
    right_disparity = select_best_disparity(right_aggregation, 64)

    imshow('left', left_disparity)
    imshow('right', right_disparity)
    waitKey()


if __name__ == "__main__":
    sgm()
