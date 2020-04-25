from numpy import zeros, pad, uint8, uint64
from cv2 import normalize, NORM_MINMAX


def normalize_image(image):
    """
    Normalize image

    :param image: Image vector to normalize

    :return: Image normalized
    """
    return normalize(image, dst=None, alpha=0, beta=255, norm_type=NORM_MINMAX).astype(uint8)


def census_transformation(image, census_kernel_height, census_kernel_width):
    """
    Do a census transformation on image

    :param image: Image vector to do a transformation
    :param census_kernel_height: Kernel height size
    :param census_kernel_width: Kernel width size

    :return: A image with census transformation
    """

    height = image.shape[0]  # Rows
    width = image.shape[1]  # Cols
    c = int(census_kernel_height / 2)
    r = int(census_kernel_width / 2)

    census = zeros((height, width), dtype=uint64)

    for i in range(r, height - r - 1):
        for j in range(c, width - c - 1):

            ce = 0

            for x in range(-r, r):
                for y in range(-c, c):
                    if x or y:
                        ce = ce << 1
                        if image[i, j] > image[i + x, j + y]:
                            ce |= 1

            census[i, j] = ce

    return census
