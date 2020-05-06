import numpy as np
import cv2


def hamming_distance(left_census_values, right_census_values, disparity, x_offset, y_offset):
    """
    Calculate the Hamming distance

    :param left_census_values: Left census.
    :param right_census_values: Right census.
    :param disparity: Max disparity value
    :param x_offset: X offset
    :param y_offset: Y offset

    :return: The hamming distance.
    """

    height = left_census_values.shape[0]
    width = left_census_values.shape[1]

    left_cost_volume = np.zeros(
        shape=(height, width, disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(
        shape=(height, width, disparity), dtype=np.uint32)

    lcensus = np.zeros(shape=(height, width), dtype=np.int64)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)

    for d in range(0, disparity):
        rcensus[:, (x_offset + d):(width - x_offset)
                ] = right_census_values[:, x_offset:(width - d - x_offset)]
        left_xor = np.int64(np.bitwise_xor(
            np.int64(left_census_values), rcensus))
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        left_cost_volume[:, :, d] = left_distance

        lcensus[:, x_offset:(
            width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        right_xor = np.int64(np.bitwise_xor(
            np.int64(right_census_values), lcensus))
        right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(right_xor == 0):
            tmp = right_xor - 1
            mask = right_xor != 0
            right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
            right_distance[mask] = right_distance[mask] + 1
        right_cost_volume[:, :, d] = right_distance

        left_cost = np.array(left_distance, np.uint8)
        right_cost = np.array(right_distance, np.uint8)

    return left_cost_volume, right_cost_volume
