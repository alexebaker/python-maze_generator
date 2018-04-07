import numpy as np
import matplotlib.pyplot as plt

from numpy.random import random_integers as randint


ACRE_WIDTH_FT = 208.71
ACRE_HEIGHT_FT = 208.71


class Field:
    _field = None

    def __init__(self, size=(4, 2), diameter=10, complexity=0.75, density=0.75):
        height = int((size[0] * ACRE_HEIGHT_FT) / diameter)
        width = int((size[1] * ACRE_WIDTH_FT) / diameter)
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        self.complexity = int(complexity * (5 * (shape[0] + shape[1])))
        self.density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        self._field = np.zeros(shape, dtype=bool)
        return

    def display(self):
        plt.imshow(self._field, cmap=plt.cm.binary, interpolation='nearest')
        plt.show()
        return

    def generate(self):
        shape = self._field.shape

        self._field[0, :] = 1
        self._field[-1, :] = 1
        self._field[:, 0] = 1
        self._field[:, -1] = 1

        for i in range(self.density):
            x, y = randint(0, shape[1] // 2) * 2, randint(0, shape[0] // 2) * 2
            self._field[y, x] = 1
            for j in range(self.complexity):
                neighbors = []
                if x > 1:
                    neighbors.append((y, x - 2))
                if x < shape[1] - 2:
                    neighbors.append((y, x + 2))
                if y > 1:
                    neighbors.append((y - 2, x))
                if y < shape[0] - 2:
                    neighbors.append((y + 2, x))
                if neighbors:
                    y_, x_ = neighbors[randint(0, len(neighbors) - 1)]
                    if self._field[y_, x_] == 0:
                        self._field[y_, x_] = 1
                        self._field[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        return
