class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        """
        Represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
        :param direction: (x, y) for cardinal direction.
        :param name: Common name of said direction.
        """
        self.direction = direction
        self.name = name
