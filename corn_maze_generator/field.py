import math
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import random_integers as randint
from queue import Queue


ACRE_WIDTH_FT = math.sqrt(43560)
ACRE_HEIGHT_FT = ACRE_WIDTH_FT


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

        for _ in range(self.density):
            x, y = randint(0, shape[1] // 2) * 2, randint(0, shape[0] // 2) * 2
            self._field[y, x] = 1
            for _ in range(self.complexity):
                neighbors = []
                if x > 1:
                    neighbors.append((y, x-2))
                if x < shape[1] - 2:
                    neighbors.append((y, x+2))
                if y > 1:
                    neighbors.append((y-2, x))
                if y < shape[0] - 2:
                    neighbors.append((y+2, x))
                if neighbors:
                    y_, x_ = neighbors[randint(0, len(neighbors) - 1)]
                    if self._field[y_, x_] == 0:
                        self._field[y_, x_] = 1
                        self._field[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        self.finish()
        return

    def finish(self):
        shape = self._field.shape
        tmp = np.where(self._field == 1)
        start_points = [(r, c) for (r, c) in zip(tmp[0], tmp[1])
                        if ((r == 0 and c < shape[1]-1) or (c == 0 and r < shape[0]-1))]

        for p in start_points:
            queue = Queue()
            meta = {}
            visited = np.zeros(self._field.shape, dtype=bool)
            queue.put(p)

            while not queue.empty():
                root = queue.get()

                for neighbor in self._get_neighbors(root):
                    if visited[neighbor]:
                        continue

                    if neighbor not in queue:
                        meta[neighbor] = root
                        queue.put(neighbor)

                visited[root] = 1
        return

    def _get_neighbors(self, idx):
        neighbors = []
        row, col = idx
        if row-1 >= 0:
            neighbors.append((row-1, col))
        if row+1 < self._field.shape[0]:
            neighbors.append((row+1, col))
        if col-1 >= 0:
            neighbors.append((row, col-1))
        if col+1 < self._field.shape[1]:
            neighbors.append((row, col+1))
        return [n for n in neighbors if not self._field[n]]
