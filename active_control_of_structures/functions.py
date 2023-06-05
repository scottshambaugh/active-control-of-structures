# functions.py
'''
This file holds the functions from Appendix A "Matlab Functions".

Reference:
Gawronski, W., "Advanced structural dynamics and active control of structures".
New York, NY: Springer New York, 2004.

'''

import numpy as np
from scipy.linalg import eig, inv


def modal1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function determines the modal representation 1 (am, bm, cm)
    given a generic state-space representation (a, b, c)
    and the transformation r to the modal representation
    such that am = inv(r) * a * r, bm = inv(r) * b, and cm = c * r

    Parameters
    ----------
    a : np.ndarray
        State matrix A.
    b : np.ndarray
        State matrix B.
    c : np.ndarray
        State matrix C.

    Returns
    -------
    r : np.ndarray
        Transformation matrix.
    am : np.ndarray
        Modal state matrix A.
    bm : np.ndarray
        Modal state matrix B.
    cm : np.ndarray
        Modal state matrix C.
    """

    # transformation to complex-diagonal form
    an, v = eig(a)
    An = np.diag(an)
    bn = inv(v) @ b
    cn = c @ v

    # transformation to modal form 1
    i = np.where(np.imag(an))[0]
    index = i[::2]
    t = np.eye(len(an), dtype=complex)
    z = np.zeros(len(an), dtype=complex)

    if len(index) == 0:
        am = An
        bm = bn
        cm = cn
        r = t

    else:
        for i in index:
            om = abs(An[i, i])
            z[i] = -np.real(An[i, i]) / abs(An[i, i])
            j = 1j
            t[i: i+2, i: i+2] = np.array([[z[i] - j * np.sqrt(1 - z[i]**2), 1],
                                          [z[i] + j * np.sqrt(1 - z[i]**2), 1]])

        # modal form 1
        am = np.real(inv(t) @ An @ t)
        bm = np.real(inv(t) @ bn)
        cm = np.real(cn @ t)
        beta = 1
        s = np.eye(len(an), dtype=complex)

        for i in index:
            alpha = -beta * bm[i+1] / bm[i]
            s[i, i] = alpha + 2 * z[i] * beta
            s[i, i+1] = beta
            s[i+1, i] = -beta
            s[i+1, i+1] = alpha

        am = inv(s) @ am @ s
        bm = inv(s) @ bm
        cm = cm @ s

        # the transformation
        r = v @ t @ s

    return r, am, bm, cm


def modal2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function determines the modal representation 2 (am, bm, cm)
    given a generic state-space representation (a, b, c)
    and the transformation r to the modal representation
    such that am = inv(r) * a * r, bm = inv(r) * b, and cm = c * r

    Parameters
    ----------
    a : np.ndarray
        State matrix A.
    b : np.ndarray
        State matrix B.
    c : np.ndarray
        State matrix C.

    Returns
    -------
    r : np.ndarray
        Transformation matrix.
    am : np.ndarray
        Modal state matrix A.
    bm : np.ndarray
        Modal state matrix B.
    cm : np.ndarray
        Modal state matrix C.
    """

    # transformation to complex-diagonal form
    an, v = eig(a)
    An = np.diag(an)
    bn = inv(v) @ b
    cn = c @ v

    # transformation to modal form 2
    i = np.where(np.imag(an))[0]
    index = i[::2]
    j = 1j
    t = np.eye(len(an), dtype=complex)

    if len(index) == 0:
        am = an
        bm = bn
        cm = cn
        r = t
    else:
        for i in index:
            t[i: i+2, i: i+2] = np.array([[j, 1], [-j, 1]])

        # modal form 2
        am = np.real(inv(t) @ An @ t)
        bm = np.real(inv(t) @ bn)
        cm = np.real(cn @ t)

    # the transformation
    r = v @ t

    return r, am, bm, cm


def modal1m(om: np.ndarray, z: np.ndarray, mm: np.ndarray, phi: np.ndarray,
            b: np.ndarray, cq: np.ndarray, cv: np.ndarray, coord: int) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Determines the modal form 1 (am, bm, cm) from modal data.

    In the parameters below, n is the number of modes and nd is the number of
    degrees of freedom.

    Parameters
    ----------
    om : np.ndarray
        Modal frequencies (nx1).
    z : np.ndarray
        Modal damping (nx1).
    mm : np.ndarray
        Modal mass matrix (ndxn).
    phi : np.ndarray
        Modal matrix (ndxn).
    b : np.ndarray 
        Input matrix (ndxs).
    cq : np.ndarray
        Displacement output matrix (rxnd).
    cv : np.ndarray
        Rate output matrix (rxnd).
    coord : int
        If coord = 0, state-space representation is in modal coordinates.
        If coord = 1, modal state-space representation.

    Returns
    -------
    am : np.ndarray
        Modal state matrix A.
    bm : np.ndarray
        Modal state matrix B.
    cm : np.ndarray
        Modal state matrix C.
    """
    # Arranging input data
    mm = np.diag(mm)
    om = np.diag(om)
    z = np.diag(z)
    c = np.concatenate((cq, cv), axis=1)

    # Modal input and output matrices
    bm = inv(mm) @ phi.T @ b
    cmq = cq @ phi
    cmv = cv @ phi

    # Representation in modal coordinates
    Am = np.block([[0 * om, om], [-om, -2 * z @ om]])
    Bm = np.vstack([0 * bm, bm])
    Cm = np.block([cmq @ inv(om), cmv])

    if coord == 0:
        # Representation in modal coordinates
        am = Am
        bm = Bm
        cm = Cm
    else:
        # Modal representation
        n = max(Am.shape) // 2
        ind = np.empty(2 * n, dtype=int)
        for i in range(n):
            ind[2 * i] = i
            ind[2 * i + 1] = i + n
        am = Am[ind, ind]
        bm = Bm[ind, :]
        cm = Cm[:, ind]

    return am, bm, cm


def modal2m(om: np.ndarray, z: np.ndarray, mm: np.ndarray, phi: np.ndarray,
            b: np.ndarray, cq: np.ndarray, cv: np.ndarray, coord: int) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Determines the modal form 2 (am, bm, cm) from modal data.

    In the parameters below, n is the number of modes and nd is the number of
    degrees of freedom.

    Parameters
    ----------
    om : np.ndarray
        Modal frequencies (nx1).
    z : np.ndarray
        Modal damping (nx1).
    mm : np.ndarray
        Modal mass matrix (ndxn).
    phi : np.ndarray
        Modal matrix (ndxn).
    b : np.ndarray 
        Input matrix (ndxs).
    cq : np.ndarray
        Displacement output matrix (rxnd).
    cv : np.ndarray
        Rate output matrix (rxnd).
    coord : int
        If coord = 0, state-space representation is in modal coordinates.
        If coord = 1, modal state-space representation.

    Returns
    -------
    am : np.ndarray
        Modal state matrix A.
    bm : np.ndarray
        Modal state matrix B.
    cm : np.ndarray
        Modal state matrix C.
    """
    # Arranging input data
    mm = np.diag(mm)
    om = np.diag(om)
    z = np.diag(z)
    c = np.concatenate((cq, cv), axis=1)

    # Modal input and output matrices
    bm = inv(mm) @ phi.T @ b
    cmq = cq @ phi
    cmv = cv @ phi

    # Representation in modal coordinates
    Am = np.block([[-z @ om, om], [-om, -z @ om]])
    Bm = np.vstack([0 * bm, bm])
    Cm = np.block([cmq @ inv(om) - cmv @ z, cmv])

    if coord == 0:
        # Representation in modal coordinates
        am = Am
        bm = Bm
        cm = Cm
    else:
        # Modal representation
        n = max(Am.shape) // 2
        ind = np.empty(2 * n, dtype=int)
        for i in range(n):
            ind[2 * i] = i
            ind[2 * i + 1] = i + n
        am = Am[ind, ind]
        bm = Bm[ind, :]
        cm = Cm[:, ind]

    return am, bm, cm


def modal1n(m, damp, k, b, cq, cv, n, coord):
    """
    Determines the modal form 1 (am, bm, cm) from nodal data.

    In the parameters below, n is the number of modes and nd is the number of
    degrees of freedom.

    Parameters
    ----------
    m : np.ndarray
        Mass matrix (ndxnd).
    damp : np.ndarray
        Damping matrix (ndxnd).
    k : np.ndarray
        Stiffness matrix (ndxnd).
    b : np.ndarray 
        Input matrix (ndxs).
    cq : np.ndarray
        Displacement output matrix (rxnd).
    cv : np.ndarray
        Rate output matrix (rxnd).
    coord : int
        If coord = 0, state-space representation is in modal coordinates.
        If coord = 1, modal state-space representation.

    Returns
    -------
    am : np.ndarray
        Modal state matrix A.
    bm : np.ndarray
        Modal state matrix B.
    cm : np.ndarray
        Modal state matrix C.
    """
    # modal matrix:
    om2, phi = eig(k, m)
    phi = phi[:, :n]

    # natural frequency matrix
    om = np.sqrt(om2)
    Om = np.diag(om)

    # modal mass, stiffness and damping matrices:
    mm = phi.T @ m @ phi
    km = phi.T @ k @ phi
    dm = phi.T @ damp @ phi
    z = 0.5 * inv(mm) @ dm @ inv(Om)

    # input and output matrices
    c = np.concatenate((cq, cv), axis=1)
    bm = inv(mm) @ phi.T @ b
    cmq = cq @ phi
    cmv = cv @ phi

    # Representation in modal coordinates
    Am = np.block([[0 * Om, Om], [-Om, -2 * z @ Om]])
    Bm = np.vstack([0 * bm, bm])
    Cm = np.block([cmq @ inv(Om), cmv])

    if coord == 0:
        # Representation in modal coordinates
        am = Am
        bm = Bm
        cm = Cm
    else:
        # Modal representation
        n = max(Am.shape) // 2
        ind = np.empty(2 * n, dtype=int)
        for i in range(n):
            ind[2 * i] = i
            ind[2 * i + 1] = i + n
        am = Am[ind, ind]
        bm = Bm[ind, :]
        cm = Cm[:, ind]

    return am, bm, cm


def modal2n(m, damp, k, b, cq, cv, n, coord):
    """
    Determines the modal form 1 (am, bm, cm) from nodal data.

    In the parameters below, n is the number of modes and nd is the number of
    degrees of freedom.

    Parameters
    ----------
    m : np.ndarray
        Mass matrix (ndxnd).
    damp : np.ndarray
        Damping matrix (ndxnd).
    k : np.ndarray
        Stiffness matrix (ndxnd).
    b : np.ndarray 
        Input matrix (ndxs).
    cq : np.ndarray
        Displacement output matrix (rxnd).
    cv : np.ndarray
        Rate output matrix (rxnd).
    coord : int
        If coord = 0, state-space representation is in modal coordinates.
        If coord = 1, modal state-space representation.

    Returns
    -------
    am : np.ndarray
        Modal state matrix A.
    bm : np.ndarray
        Modal state matrix B.
    cm : np.ndarray
        Modal state matrix C.
    """
    # modal matrix:
    om2, phi = eig(k, m)
    phi = phi[:, :n]

    # natural frequency matrix
    om = np.sqrt(om2)
    Om = np.diag(om)

    # modal mass, stiffness and damping matrices:
    mm = phi.T @ m @ phi
    km = phi.T @ k @ phi
    dm = phi.T @ damp @ phi
    z = 0.5 * inv(mm) @ dm @ inv(Om)

    # input and output matrices
    c = np.concatenate((cq, cv), axis=1)
    bm = inv(mm) @ phi.T @ b
    cmq = cq @ phi
    cmv = cv @ phi

    # Representation in modal coordinates
    Am = np.block([[-z @ om, om], [-om, -z @ om]])
    Bm = np.vstack([0 * bm, bm])
    Cm = np.block([cmq @ inv(om) - cmv @ z, cmv])

    if coord == 0:
        # Representation in modal coordinates
        am = Am
        bm = Bm
        cm = Cm
    else:
        # Modal representation
        n = max(Am.shape) // 2
        ind = np.empty(2 * n, dtype=int)
        for i in range(n):
            ind[2 * i] = i
            ind[2 * i + 1] = i + n
        am = Am[ind, ind]
        bm = Bm[ind, :]
        cm = Cm[:, ind]

    return am, bm, cm

