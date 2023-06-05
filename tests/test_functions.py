# test_functions.py

import numpy as np
from numpy.linalg import inv
from active_control_of_structures.functions import *

def test_modal1():
    a = np.array([[0.1, -1], [1, 0.2]])
    b = np.array([2, 3])
    c = np.array([4, 5])
    r, am, bm, cm = modal1(a, b, c)

    assert not np.allclose(r, np.eye(2))  # make sure we have complex eigenvalues
    assert np.allclose(am, inv(r) @ a @ r)
    assert np.allclose(bm, inv(r) @ b)
    assert np.allclose(cm, c @ r)


def test_modal2():
    a = np.array([[0.1, -1], [1, 0.2]])
    b = np.array([2, 3])
    c = np.array([4, 5])
    r, am, bm, cm = modal2(a, b, c)

    assert not np.allclose(r, np.eye(2))  # make sure we have complex eigenvalues
    assert np.allclose(am, inv(r) @ a @ r)
    assert np.allclose(bm, inv(r) @ b)
    assert np.allclose(cm, c @ r)

def test_modal1m():
    # TODO
    assert True

def test_modal2m():
    # TODO
    assert True

def test_modal1n():
    # TODO
    assert True

def test_modal2n():
    # TODO
    assert True
