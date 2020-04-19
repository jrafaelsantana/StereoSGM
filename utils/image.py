from cv2 import imread, GaussianBlur, imshow, waitKey
from numpy import zeros
from math import ceil


def load_images(left_name, right_name):
    """
    Read image pair

    :param left_name: Name of the left image.
    :param right_name: Name of the right image.

    :return: Left and right images.
    """

    left = imread(left_name, 0)
    right = imread(right_name, 0)

    return left, right


def blur_image(image, blur_size):
    """
    Blur image pair

    :param left_name: Left image.
    :param right_name: Right image.

    :return: Left and right images with blur.
    """

    left = GaussianBlur(left, blur_size, 0, 0)
    right = GaussianBlur(right, blur_size, 0, 0)

    return left, right


def census_transformation(image, census_kernel_height, census_kernel_width):
    """
    Do a census transformation on image

    :param image: Image vector to do a transformation
    :param census_kernel_height: Kernel height size
    :param census_kernel_width: Kernel width size

    :return: A image with census transformation
    """

    height = image.shape[0]
    width = image.shape[1]

    y_offset = int(census_kernel_height / 2)
    x_offset = int(census_kernel_width / 2)

    ceil_y_offset = ceil(census_kernel_height/2)
    ceil_x_offset = ceil(census_kernel_width/2)

    census = zeros((height-ceil_y_offset, width-ceil_x_offset), dtype='uint8')
    cp = image[y_offset:height-y_offset, x_offset:width-x_offset]

    offsets = [(u, v) for v in range(census_kernel_height)
               for u in range(census_kernel_width) if not u == 1 == v]

    for u, v in offsets:
        census = (census << 1) | (
            image[v: v+height-ceil_y_offset, u: u+width-ceil_x_offset] >= cp)

    return census
