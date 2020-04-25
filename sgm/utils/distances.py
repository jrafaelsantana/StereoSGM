from numpy import sum, unpackbits, bitwise_xor


def hamming_distance(left, right):
    """
    Calculate the Hamming distance

    :param a: First array.
    :param b: Second array.

    :return: The hamming distance between a and b.
    """

    return sum(unpackbits(bitwise_xor(left, right), axis=1), axis=1).reshape(left.shape[0], left.shape[1])
