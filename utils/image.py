from numpy import zeros, pad, uint8, float64
from cv2 import normalize, NORM_MINMAX, GaussianBlur, medianBlur, imread


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


def blur_image(left, right, blur_size):
    """
    Blur image pair

    :param left: Left image.
    :param right: Right image.
    :param blur_size: Blur kernel size

    :return: Left and right images with blur.
    """

    left = GaussianBlur(left, (blur_size, blur_size), 0, 0)
    right = GaussianBlur(right, (blur_size, blur_size), 0, 0)

    return left, right


def normalize_image(image, max_disparity):
    """
    Normalize image

    :param image: Image vector to normalize

    :return: Image normalized
    """

    # return normalize(image, dst=None, alpha=0, beta=255, norm_type=NORM_MINMAX, dtype=uint8)
    return 255.0 * image / max_disparity


def census_transformation(image, census_kernel):
    """
    Do a census transformation on image

    :param image: Image vector to do a transformation
    :param census_kernel: Kernel size

    :return: A image with census transformation
    """

    height = image.shape[0]  # Rows
    width = image.shape[1]  # Cols
    c = int(census_kernel / 2)

    census = zeros((height, width), dtype=float64)

    for i in range(c, height - c - 1):
        for j in range(c, width - c - 1):

            ce = 0

            for x in range(-c, c):
                for y in range(-c, c):
                    if x or y:
                        ce = ce << 1
                        if image[i, j] > image[i + x, j + y]:
                            ce |= 1

            census[i, j] = ce

    return census


def median_filter(left, right, blur_size):
    left = medianBlur(left, blur_size)
    right = medianBlur(right, blur_size)

    return left, right
