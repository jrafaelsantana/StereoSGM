from cv2 import imread, GaussianBlur


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
