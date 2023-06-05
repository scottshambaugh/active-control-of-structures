# test_structural_parameters.py

from active_control_of_structures.structural_parameters import *

def test_truss_2d():
    # Does not test the values of the matrices, just that they are returned.
    A, M = truss_2d()
    assert True

def test_clamped_beam():
    # Does not test the values of the matrices, just that they are returned.
    A, M = clamped_beam()
    assert True

def test_deep_space_network_antenna():
    # Does not test the values of the matrices, just that they are returned.
    A, B, C = deep_space_network_antenna()
    assert True