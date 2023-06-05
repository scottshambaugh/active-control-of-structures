# structural_parameters.py
"""
This file holds the data from Appendix C "Structural Parameters".
"""

import numpy as np
from scipy.linalg import block_diag


def truss_2d() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the mass and stiffness matrices for the 2D truss structure.
    """

    M1 = np.diag([0.41277, 0.41277, 0.41277, 0.41277, 0.41277, 0.41277, 0.23587, 0.23587])
    O = np.zeros((8, 8))
    M = np.block([[M1, O], [O, M1]])

    K1 = np.array([[3.024, -1, 0, 0, 0, 0, 0, 0],
                   [0, 1.909, 0, 0, 0, 0, 0, 0],
                   [-1, 0, 3.024, 0, -1, 0, 0, 0],
                   [0, 0, 0, 1.909, 0, 0, 0, 0],
                   [0, 0, -1, 0, 3.024, 0, -1, 0],
                   [0, 0, 0, 0, 0, 1.909, 0, 0],
                   [0, 0, 0, 0, -1, 0, 1.512, -0.384],
                   [0, 0, 0, 0, 0, 0, -0.384, 1.621]]) * 1e6                  

    K2 = np.array([[0, 0, -0.512, -0.384, 0, 0, 0, 0],
                   [0, -1.333, -0.384, -0.288, 0, 0, 0, 0],
                   [-0.512, 0.384, 0, 0, -0.512, -0.384, 0, 0],
                   [0.384, -0.288, 0, -1.333, -0.384, -0.288, 0, 0],
                   [0, 0, -0.512, 0.384, 0, 0, -0.512, -0.384],
                   [0, 0, 0.384, -0.288, 0, -1.333, -0.384, -0.288],
                   [0, 0, 0, 0, -0.512, 0.384, 0, 0],
                   [0, 0, 0, 0, 0.384, -0.288, 0, -1.333]]) * 1e6
    
    K = np.block([[K1, K2], [K2.T, K1]])

    return M, K


def clamped_beam() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the mass and stiffness matrices for the clamped beam structure.
    """
    
    M1 = np.diag([0.7850, 0.7850, 6.5417]) * 1e-4
    M = np.kron(np.eye(14), M1)

    K1 = np.diag([4.200, 0.010, 0.336]) * 1e5
    K2 = np.array([[-2.100, 0, 0],
                   [0, -0.005, 0.025],
                   [0, -0.025, 0.084]]) * 1e5
    
    K = np.kron(np.eye(14), K1) + np.kron(np.eye(14, k=1), K2) + np.kron(np.eye(14, k=-1), K2.T)

    return M, K


def deep_space_network_antenna() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the state space representation of the Deep Space Network antenna.
    """

    A1 = np.array([[0, 0], [0, -1.104067]])
    A2 = np.array([[-0.348280, 10.099752], [10.099752, -0.348280]])
    A3 = np.array([[-0.645922, 12.561336], [-12.561336, -0.645922]])
    A4 = np.array([[-0.459336, 13.660350], [-13.660350, -0.459336]])
    A5 = np.array([[-0.934874, 18.937362], [-18.937362, -0.934874]])
    A6 = np.array([[-0.580288, 31.331331], [31.331331, -0.580288]])
    A7 = np.array([[-0.842839, 36.140547], [-36.140547, -0.842839]])
    A8 = np.array([[-0.073544, 45.862202], [-45.862202, -0.073544]])
    A9 = np.array([[-3.569534, 48.508185], [-48.508185, -3.569534]])

    A = block_diag(A1, A2, A3, A4, A5, A6, A7, A8, A9)

    B = np.array([
        1.004771,
        -0.206772,
        -0.093144,
        0.048098,
        0.051888,
        1.292428,
        -0.024689,
        0.245969,
        -0.234201,
        0.056769,
        0.540327,
        -0.298787,
        -0.329058,
        -0.012976,
        -0.038636,
        -0.031413,
        -0.115836,
        0.421496
    ]).T

    C1 = np.array([1.004771, -0.204351, 0.029024, -0.042791, -0.322601, -0.545963])
    C2 = np.array([-0.098547, -0.070542, 0.113774, 0.030378, 0.058073, 0.294883])
    C3 = np.array([0.110847, -0.109961, -0.022496, -0.009963, 0.059871, -0.198378])

    C = np.concatenate((C1, C2, C3))

    return A, B, C
