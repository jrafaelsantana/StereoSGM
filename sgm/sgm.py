from paths import Paths
from utils import census_transformation, hamming_distance, normalize_image
import numpy as np

from cv2 import imread, GaussianBlur, imshow, waitKey

"""
    Semi-global matching

    Steps:
        1- Compute costs (Census transformation and Hamming distance)
        2- Compute left and right cost volume
        3- Compute left and right aggregation volume
        4- Select best disparity
        5- Apply median filter
        6- Evaluate
"""


def cost_correspondence(a, b, disparity):
    """
    Compute cost difference between left and right census tranformed images.

    :param a: First census image
    :param b: Seconf census image
    :param disparity: Disparity int

    :return: Array [H, W] with costs
    """
    height = a.shape[0]
    width = a.shape[1]

    costs = np.full(shape=(height, width), fill_value=0)

    for col in range(disparity, width):
        costs[:, col] = hamming_distance(
            a[:, col:col + 1],
            b[:, col - disparity:col - disparity + 1]
        ).reshape(a.shape[0])

    return costs


def compute_costs(left, right, kernel_size_census, max_disparity):
    """
    Matching cost based on census transform and hamming distance.

    :param left: Left image.
    :param right: Right image.
    :param kernel_size_census: Dictionary with height and width of census kernel size.
    :param max_disparity: Maximum disparity.

    :return: An array [D, H, W] with the matching costs (Height, width and disparity).
    """

    height = left.shape[0]
    width = left.shape[1]
    census_kernel_height = kernel_size_census[0]
    census_kernel_width = kernel_size_census[1]

    y_offset = int(census_kernel_height / 2)
    x_offset = int(census_kernel_width / 2)

    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    left_cost_volume = np.zeros(
        shape=(height, width, max_disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(
        shape=(height, width, max_disparity), dtype=np.uint32)

    # Census transformation

    left_census_values = census_transformation(
        left, census_kernel_height, census_kernel_width)
    right_census_values = census_transformation(
        right, census_kernel_height, census_kernel_width)

    # Hamming distance

    left_cost = []
    right_cost = []

    for disparity in range(0, max_disparity):
        left_cost.append(normalize_image(cost_correspondence(
            right_census_values, left_census_values, disparity)))
        right_cost.append(normalize_image(cost_correspondence(
            left_census_values, right_census_values, disparity)))

    left_cost = np.asarray(left_cost)
    right_cost = np.asarray(right_cost)

    return left_cost, right_cost


def sgm():
    left = imread('im2.png', 0)
    right = imread('im6.png', 0)

    left = GaussianBlur(left, (3, 3), 0, 0)
    right = GaussianBlur(right, (3, 3), 0, 0)

    compute_costs(left, right, (5, 5), 64)


if __name__ == "__main__":
    sgm()
