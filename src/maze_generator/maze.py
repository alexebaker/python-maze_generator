# -*- coding: utf-8 -*-

from __future__ import division

import math
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import random_integers as randint


ACRE_WIDTH_FT = math.sqrt(43560)
ACRE_HEIGHT_FT = ACRE_WIDTH_FT


class Maze(object):
    """Base maze object.

    Args:
        size ((int, int)): Tuple descriding the dimensions of the maze (in acres).
        diameter (int): Size of the isles in the maze.
        complexity (float): 0-1, where 1 is more complex
        density (float): 0-1, where 1 is more dense

    Example:
        maze = Maze(size=(4, 2), diameter=10, complexity=0.8, density=0.2)
        maze.generate() # create the maze
        maze.find_solution() # find a start and end point with the solution
        maze.display() # display the maze using matplotlib
    """
    _maze = None
    _solution = None

    def __init__(self, size=(4, 2), diameter=10, complexity=0.75, density=0.75):
        height = (size[0] * ACRE_HEIGHT_FT) // diameter
        width = (size[1] * ACRE_WIDTH_FT) // diameter
        shape = (int((height // 2) * 2 + 1) + 2, int((width // 2) * 2 + 1) + 2)

        self.complexity = int(complexity * (5 * (shape[0] + shape[1])))
        self.density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        self._maze = np.zeros(shape, dtype=bool)
        self._solution = []

        edge_points = []
        for row in range(0, self._maze.shape[0]):
            edge_points.append((row, 0))
            edge_points.append((row, self._maze.shape[1]-1))
        for col in range(0, self._maze.shape[1]):
            edge_points.append((0, col))
            edge_points.append((self._maze.shape[0]-1, col))
        self.edge_points = edge_points
        return

    def display(self):
        """Displays the maze using matplotlib."""
        maze = np.zeros((self._maze.shape[0], self._maze.shape[1], 3))
        maze[self._maze == 0] = [1, 1, 1]
        for row, col in self._solution:
            maze[row, col] = [1, 0, 0]
        plt.imshow(maze, interpolation='nearest')
        plt.show()
        return

    def generate(self):
        """Generates the maze.

        Uses the following algorithm for maze generation:
        https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """
        shape = self._maze.shape

        self._maze[0, :] = 1
        self._maze[-1, :] = 1
        self._maze[:, 0] = 1
        self._maze[:, -1] = 1

        for _ in range(self.density):
            x, y = randint(0, shape[1] // 2) * 2, randint(0, shape[0] // 2) * 2
            self._maze[y, x] = 1
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
                    if self._maze[y_, x_] == 0:
                        self._maze[y_, x_] = 1
                        self._maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        edge_points = []
        for row in range(1, self._maze.shape[0]-1):
            edge_points.append((row, 1))
            edge_points.append((row, self._maze.shape[1]-2))
        for col in range(1, self._maze.shape[1]):
            edge_points.append((1, col))
            edge_points.append((self._maze.shape[0]-2, col))
        for point in edge_points:
            if not self._maze[point]:
                self._maze[point] = True
                if not self._is_connected():
                    self._maze[point] = False
        return

    def find_solution(self):
        """Finds a start and endpoint and finds the solution between them.

        This method compares all possible start and end points, and pics the two
        which give the longest, shortest, path.
        """
        for p in self.edge_points:
            queue = BFSQueue()
            meta = {p: None}
            visited = np.zeros(self._maze.shape, dtype=bool)
            queue.put(p)

            while not queue.empty():
                root = queue.get()

                for neighbor in self._get_neighbors(root):
                    if not self._maze[neighbor]:
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

    def is_connected(self):
        """Test is the maze is connected."""
        queue = BFSQueue()
        start_point = np.where(self._maze == 0)[0][0], np.where(self._maze == 0)[1][0]
        visited = np.ones(self._maze.shape, dtype=bool)
        queue.put(start_point)

        while not queue.empty():
            root = queue.get()

            for neighbor in self._get_neighbors(root):
                if not self._maze[neighbor]:
                    if not visited[neighbor]:
                        continue

                    if neighbor not in queue:
                        queue.put(neighbor)
            visited[root] = False
        return np.array_equal(self._maze, visited)

    def _get_solution(self, point, meta):
        solution = []
        while point:
            solution.append(point)
            point = meta[point]
        return solution

    def _get_neighbors(self, point):
        neighbors = []
        row, col = point
        if row-1 >= 0:
            neighbors.append((row-1, col))
        if row+1 < self._maze.shape[0]:
            neighbors.append((row+1, col))
        if col-1 >= 0:
            neighbors.append((row, col-1))
        if col+1 < self._maze.shape[1]:
            neighbors.append((row, col+1))
        return neighbors


class BFSQueue(object):
    """Simple queue implementation for use with the BFS algorithm. """
    def __init__(self):
        self._queue = []
        return

    def get(self):
        """Remove and return the next item in the queue.

        Returns:
            object: next item in the queue.

        Raises:
            BFSQueueEmpty: If the queue is empty and there is nothing to get.
        """
        if not self.is_empty():
            item = self._queue[0]
            self._queue = self._queue[1:]
            return item
        raise BFSQueueEmpty("Cannot perform 'get' on an empty queue.")

    def put(self, item):
        """Insert an item to the back of the queue.

        Args:
            item (object): item to insert into the queue.
        """
        self._queue.append(item)
        return

    def is_empty(self):
        """Checks whether the queue is empty or not.

        Returns:
            bool: True if there is nothing in the queue, False otherwise
        """
        return not bool(self._queue)

    def __contains__(self, item):
        return item in self._queue

    def __hash__(self):
        return self._queue.__hash__()


class BFSQueueEmpty(Exception):
    pass
