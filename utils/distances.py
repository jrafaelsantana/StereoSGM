from numpy import count_nonzero


def hamming_distance(a, b):
    """
    Calculate the Hamming distance

    :param a: First array.
    :param b: Second array.

    :return: The hamming distance between a and b.
    """

    return count_nonzero(a != b)
