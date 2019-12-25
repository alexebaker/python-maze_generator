# -*- coding: utf-8 -*-

import pytest

from maze_generator.maze import BFSQueue
from maze_generator.maze import BFSQueueEmpty


def test_empty():
    q = BFSQueue()
    assert q.is_empty()
    q.put(1)
    assert not q.is_empty()
    return


def test_put():
    q = BFSQueue()
    q.put(1)
    assert [1] == q._queue
    return


def test_get():
    q = BFSQueue()
    q._queue = [1, 2]
    assert 1 == q.get()
    assert [2] == q._queue
    assert 2 == q.get()
    assert q.is_empty()

    with pytest.raises(BFSQueueEmpty):
        q.get()
    return


def test_contains():
    q = BFSQueue()
    assert 1 not in q
    q.put(1)
    assert 1 in q
    return
