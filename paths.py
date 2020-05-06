from direction import Direction


class Paths:
    def __init__(self):
        """
        Represent the relation between the directions.
        """

        self.N = Direction(direction=(0, -1), name='north')
        self.NE = Direction(direction=(1, -1), name='north-east')
        self.E = Direction(direction=(1, 0), name='east')
        self.SE = Direction(direction=(1, 1), name='south-east')
        self.S = Direction(direction=(0, 1), name='south')
        self.SW = Direction(direction=(-1, 1), name='south-west')
        self.W = Direction(direction=(-1, 0), name='west')
        self.NW = Direction(direction=(-1, -1), name='north-west')

        self.paths = [self.N, self.NE, self.E,
                      self.SE, self.S, self.SW, self.W, self.NW]
        self.size = len(self.paths)
        self.effective_paths = [
            (self.E,  self.W), (self.SE, self.NW), (self.S, self.N), (self.SW, self.NE)]
