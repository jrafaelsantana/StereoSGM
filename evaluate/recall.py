from cv2 import imread
import numpy as np


def recall(disparity, gt, maximum_disparity):
    """
    Computes the recall of the disparity map.

    :param disparity: Disparity image.
    :param gt: Path to ground-truth image.
    :param maximum_disparity: Maximum disparity.

    :return: Tate of correct predictions.
    """
    gt = np.float32(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))
    gt = np.int16(gt / 255.0 * float(maximum_disparity))
    disparity = np.int16(np.float32(disparity) / 255.0 *
                         float(maximum_disparity))
    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    return float(correct) / gt.size
