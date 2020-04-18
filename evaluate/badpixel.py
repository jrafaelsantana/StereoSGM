import numpy as np


def badpixel(disp_map, pfm_data, threshold=1, offset_x=-1):
    """
    Function to calculate badpixel rate. 

    :param disp_map: Generated disparity map.
    :param pfm_data: Pfm file.
    :param threshold: Threshold value (default = 1)
    :param offset_x: Ignores the first x values during the calculation. 
        If -1, consider the full disparity map (default = -1)

    :return: Bad pixel rate.
    """

    error = 0
    total = 0
    disp_map = np.squeeze(np.asarray(disp_map))
    pfm_data = np.squeeze(np.asarray(pfm_data))
    for y in range(0, pfm_data.shape[0]):
        for x in range(0, pfm_data.shape[1]):
            if not (offset_x >= 0 and x < offset_x):
                if not np.isinf(pfm_data[y][x]):
                    if abs(pfm_data[y][x]-disp_map[y][x]) > threshold:
                        error += 1
                    total += 1

    return error/total
