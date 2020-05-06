from numpy import array, reshape, zeros, abs, repeat, amin


def get_path_cost(slice, offset, p1, p2):
    """
    Finds the minimum costs in a D x M slice (where M = the number of pixels in the given direction)

    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param p1: Penalty equals 1.
    :param p2: Penalty bigger then 1.

    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = zeros(
        shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[abs(disparities - disparities.T) == 1] = p1
    penalties[abs(disparities - disparities.T) > 1] = p2

    minimum_cost_path = zeros(
        shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = repeat(previous_cost, repeats=disparity_dim,
                          axis=0).reshape(disparity_dim, disparity_dim)
        costs = amin(costs + penalties, axis=0)

        minimum_cost_path[i, :] = current_cost + costs - amin(previous_cost)

    return minimum_cost_path

def get_indices(paths, offset, dim, direction, height):
    """
    For the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
    
    :param offset: difference with the main diagonal of the cost volume.
    :param dim: number of elements along the path.
    :param direction: current aggregation direction.
    :param height: H of the cost volume.
    
    :return: arrays for the y (H dimension) and x (W dimension) indices.
    """
    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == paths.SE.direction:
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(0 + i)
            else:
                y_indices.append(0 + i)
                x_indices.append(offset + i)

        if direction == paths.SW.direction:
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(0 + i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return array(y_indices), array(x_indices)