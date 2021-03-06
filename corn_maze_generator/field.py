import math
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import random_integers as randint


ACRE_WIDTH_FT = math.sqrt(43560)
ACRE_HEIGHT_FT = ACRE_WIDTH_FT


class Field:
    _field = None

    def __init__(self, size=(4, 2), diameter=10, complexity=0.75, density=0.75):
        height = (size[0] * ACRE_HEIGHT_FT) // diameter
        width = (size[1] * ACRE_WIDTH_FT) // diameter
        shape = (int((height // 2) * 2 + 1) + 2, int((width // 2) * 2 + 1) + 2)

        self.complexity = int(complexity * (5 * (shape[0] + shape[1])))
        self.density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        self._field = np.zeros(shape, dtype=bool)
        self._solution = []

        edge_points = []
        for row in range(0, self._field.shape[0]):
            edge_points.append((row, 0))
            edge_points.append((row, self._field.shape[1]-1))
        for col in range(0, self._field.shape[1]):
            edge_points.append((0, col))
            edge_points.append((self._field.shape[0]-1, col))
        self.edge_points = edge_points
        return

    def display(self):
        field = np.zeros((self._field.shape[0], self._field.shape[1], 3))
        field[self._field == 0] = [1, 1, 1]
        for row, col in self._solution:
            field[row, col] = [1, 0, 0]
        plt.imshow(field, interpolation='nearest')
        plt.show()
        return

    def generate(self):
        """https://en.wikipedia.org/wiki/Maze_generation_algorithm"""
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
                    y_, x_ = neighbors[randint(0, len(neighbors)-1)]
                    if self._field[y_, x_] == 0:
                        self._field[y_, x_] = 1
                        self._field[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        edge_points = []
        for row in range(1, self._field.shape[0]-1):
            edge_points.append((row, 1))
            edge_points.append((row, self._field.shape[1]-2))
        for col in range(1, self._field.shape[1]):
            edge_points.append((1, col))
            edge_points.append((self._field.shape[0]-2, col))
        for point in edge_points:
            if not self._field[point]:
                self._field[point] = True
                if not self._is_connected():
                    self._field[point] = False
        return

    def find_solution(self):
        for p in self.edge_points:
            queue = BFSQueue()
            meta = {p: None}
            visited = np.zeros(self._field.shape, dtype=bool)
            queue.put(p)

            while not queue.empty():
                root = queue.get()

                for neighbor in self._get_neighbors(root):
                    if not self._field[neighbor]:
                        if visited[neighbor]:
                            continue

                        if neighbor not in queue:
                            meta[neighbor] = root
                            queue.put(neighbor)
                    elif neighbor in self.edge_points and root not in self.edge_points:
                        new_solution = self._get_solution(root, meta)
                        new_solution.append(neighbor)
                        if len(new_solution) > len(self._solution):
                            self._solution = new_solution

                visited[root] = True
        return

    def _get_solution(self, point, meta):
        solution = []
        while point:
            solution.append(point)
            point = meta[point]
        return solution

    def _is_connected(self):
        queue = BFSQueue()
        start_point = np.where(self._field == 0)[0][0], np.where(self._field == 0)[1][0]
        visited = np.ones(self._field.shape, dtype=bool)
        queue.put(start_point)

        while not queue.empty():
            root = queue.get()

            for neighbor in self._get_neighbors(root):
                if not self._field[neighbor]:
                    if not visited[neighbor]:
                        continue

                    if neighbor not in queue:
                        queue.put(neighbor)
            visited[root] = False
        return np.array_equal(self._field, visited)

    def _get_neighbors(self, point):
        neighbors = []
        row, col = point
        if row-1 >= 0:
            neighbors.append((row-1, col))
        if row+1 < self._field.shape[0]:
            neighbors.append((row+1, col))
        if col-1 >= 0:
            neighbors.append((row, col-1))
        if col+1 < self._field.shape[1]:
            neighbors.append((row, col+1))
        return neighbors


class BFSQueue:
    def __init__(self):
        self._queue = []

    def get(self):
        item = self._queue[0]
        self._queue = self._queue[1:]
        return item

    def put(self, item):
        self._queue.append(item)
        return

    def empty(self):
        return not bool(self._queue)

    def __contains__(self, item):
        return item in self._queue

    def __hash__(self):
        return self._queue.__hash__()
