# functions.py
'''
This file holds the python implementation of the functions from Appendix A
"Matlab Functions" of the reference.

Reference:
Gawronski, W., "Advanced structural dynamics and active control of structures".
New York, NY: Springer New York, 2004.

'''

import numpy as np
import control
from scipy.linalg import (eig, inv, logm, expm, sqrtm, svd, norm,
                          solve_continuous_lyapunov, solve_continuous_are)


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
    idx = an.argsort()[::-1]  # sort eigenvalues from largest to smallest
    an = an[idx]
    v = v[:, idx]

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
    idx = an.argsort()[::-1]  # sort eigenvalues from largest to smallest
    an = an[idx]
    v = v[:, idx]

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
    idx = om2.argsort()[::-1]  # sort eigenvalues from largest to smallest
    om2 = om2[idx]
    phi = phi[:, idx]
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


def modal_time_fr(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                  t1: float, t2: float, om1: float, om2: float) -> \
                    tuple[np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function finds modal representation (am, bm, cm) and transformation r
    in limited-time interval [t1, t2], and limited-frequency interval [om1, om2]

    Parameters
    ----------
    a : np.ndarray
        State matrix A.
    b : np.ndarray
        State matrix B.
    c : np.ndarray
        State matrix C.
    t1 : float
        Initial time [s].
    t2 : float
        Final time [s].
    om1 : float
        Initial frequency [rad/s].
    om2 : float
        Final frequency [rad/s].

    Returns
    -------
    am : np.ndarray
        Modal state matrix A.
    bm : np.ndarray
        Modal state matrix B.
    cm : np.ndarray
        Modal state matrix C.
    g : np.ndarray
        Hankel singular values.
    r : np.ndarray
        Transformation matrix.
    wc : np.ndarray
        Controllability grammian.
    wo : np.ndarray
        Observability grammian.    
    """

    # modal representation:
    r, am, bm, cm = modal1(a, b, c)

    # finite-frequency transformation matrix sw,
    # and finite-frequency grammians wcw and wow:
    j = 1j
    n1, n2 = am.shape
    i = np.eye(n1, dtype=complex)
    x1 = j * om1 * i + am
    x2 = np.linalg.inv(-j * om1 * i + am)
    s1 = (j / 2 / np.pi) * logm(x1 @ x2)
    x1 = j * om2 * i + am
    x2 = np.linalg.inv(-j * om2 * i + am)
    s2 = (j / 2 / np.pi) * logm(x1 @ x2)
    sw = s2 - s1

    # grammians:
    wc = solve_continuous_lyapunov(am, bm @ bm.T)  # controllability grammian
    wo = solve_continuous_lyapunov(am.T, cm.T @ cm)  # observability grammian

    # finite-frequency grammians:
    wcw = wc @ sw.conj().T + sw @ wc
    wow = sw.conj().T @ wo + wo @ sw

    # finite-time transformation matrices st1, st2, and finite time
    # and frequency grammians wcTW and woTW:
    st1 = -expm(am * t1)
    st2 = -expm(am * t2)
    wct1W = st1 @ wcw @ st1.T
    wct2W = st2 @ wcw @ st2.T
    wcTW = wct1W - wct2W
    wot1W = st1.T @ wo @ st1
    wot2W = st2.T @ wo @ st2
    woTW = wot1W - wot2W

    # sorting in descending order of the Hankel singular values:
    wc = np.real(wcTW)
    wo = np.real(woTW)
    g = np.sqrt(np.abs(np.diag(wc @ wo)))
    ind = np.argsort(-g)
    g = -g[ind]
    am = am[ind, :][:, ind]
    bm = bm[ind, :]
    cm = cm[:, ind]

    return am, bm, cm, g, r, wc, wo


def balan2(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function finds the open-loop balanced representation (Ab, Bb, Cb) so 
    that controllability (Wc) and observability (Wo) grammians are equal and
    diagonal: Wc = Wo = Gamma

    Parameters
    ----------
    A : np.ndarray
        State matrix A.
    B : np.ndarray
        State matrix B.
    C : np.ndarray
        State matrix C.

    Returns
    -------
    Ab : np.ndarray
        Balanced state matrix A.
    Bb : np.ndarray
        Balanced state matrix B.
    Cb : np.ndarray
        Balanced state matrix C.
    R : np.ndarray
        Transformation matrix to the balanced representation.
    Gamma : np.ndarray
        Hankel singular values.
    """
    # Controllability grammian
    Wc = solve_continuous_lyapunov(A, B @ B.T)

    # Observability grammian
    Wo = solve_continuous_lyapunov(A.T, C.T @ C)

    # SVD of the controllability grammian
    Uc, Sc, Vc = svd(Wc)

    # SVD of the observability grammian
    Uo, So, Vo = svd(Wo)

    Sc = sqrtm(Sc)
    So = sqrtm(So)
    P = Uc @ Sc
    Q = So @ Vo.T

    # Hankel matrix
    H = Q @ P

    # SVD of the Hankel matrix
    V, Gamma, U = svd(H)

    G1 = sqrtm(Gamma)
    R = P @ U @ inv(G1)  # transformation matrix R

    Rinv = inv(G1) @ V.T @ Q  # inverse of R

    # Balanced representation (Ab, Bb, Cb)
    Ab = Rinv @ A @ R
    Bb = Rinv @ B
    Cb = C @ R
    
    return Ab, Bb, Cb, Gamma, R


def norm_H2(om: np.ndarray, z: np.ndarray, bm: np.ndarray,
            cmq: np.ndarray, cmr: np.ndarray, cma: np.ndarray) -> np.ndarray:
    """
    This function finds an approximate H_2 norm for each mode of a structure
    with displacement, rate, and acceleration sensors

    Parameters
    ----------
    om : np.ndarray
        Vector of natural frequencies.
    z : np.ndarray
        Vector of modal damping.
    bm : np.ndarray
        Modal matrix of actuator location.
    cmq : np.ndarray
        Modal matrix of displacement sensor location.
    cmr : np.ndarray
        Modal matrix of rate sensor location.
    cma : np.ndarray
        Modal matrix of accelerometer location.

    Returns
    -------
    norm : np.ndarray
        H_2 norm.
    """
    om2 = np.diag(om ** 2)
    bb = np.diag(bm @ bm.T)
    cc = np.diag(cma.T @ cma @ om2 + cmr.T @ cmr + cmq.T @ cmq @ inv(om2))
    h = np.sqrt(bb * cc) / 2
    h = h / np.sqrt(z)
    norm = h / np.sqrt(om)
    
    return norm


def norm_Hinf(om: np.ndarray, z: np.ndarray, bm: np.ndarray,
              cmq: np.ndarray, cmr: np.ndarray, cma: np.ndarray) -> np.ndarray:
    """
    This function finds an approximate H_inf norm for each mode of a structure
    with displacement, rate, and acceleration sensors

    Parameters
    ----------
    om : np.ndarray
        Vector of natural frequencies.
    z : np.ndarray
        Vector of modal damping.
    bm : np.ndarray
        Modal matrix of actuator location.
    cmq : np.ndarray
        Modal matrix of displacement sensor location.
    cmr : np.ndarray
        Modal matrix of rate sensor location.
    cma : np.ndarray
        Modal matrix of accelerometer location.

    Returns
    -------
    norm : np.ndarray
        H_inf norm.
    """
    om2 = np.diag(om ** 2)
    bb = np.diag(bm @ bm.T)
    cc = np.diag(cma.T @ cma @ om2 + cmr.T @ cmr + cmq.T @ cmq @ inv(om2))
    h = np.sqrt(bb * cc) / 2
    h = h / z
    norm = h / om
    
    return norm


def norm_Hankel(om: np.ndarray, z: np.ndarray, bm: np.ndarray,
                cmq: np.ndarray, cmr: np.ndarray, cma: np.ndarray) -> \
                    np.ndarray:
    """
    This function finds an approximate Hankel norm for each mode of a structure
    with displacement, rate, and acceleration sensors

    Parameters
    ----------
    om : np.ndarray
        Vector of natural frequencies.
    z : np.ndarray
        Vector of modal damping.
    bm : np.ndarray
        Modal matrix of actuator location.
    cmq : np.ndarray
        Modal matrix of displacement sensor location.
    cmr : np.ndarray
        Modal matrix of rate sensor location.
    cma : np.ndarray
        Modal matrix of accelerometer location.

    Returns
    -------
    norm : np.ndarray
        Hankel norm.
    """
    om2 = np.diag(om ** 2)
    bb = np.diag(bm @ bm.T)
    cc = np.diag(cma.T @ cma @ om2 + cmr.T @ cmr + cmq.T @ cmq @ inv(om2))
    h = np.sqrt(bb * cc) / 4
    h = h / z
    norm = h / om
    
    return norm


def bal_LQG(A, B, C, Q, R, V, W):
    """
    This function finds the LQG-balanced representation (Ab, Bb, Cb)
    so that CARE (Sc) and FARE (Se) solutions are equal and diagonal:
    Sc = Se = Mu

    Parameters
    ----------
    A : np.ndarray
        State matrix A.
    B : np.ndarray
        State matrix B.
    C : np.ndarray
        State matrix C.
    Q : np.ndarray
        State weight matrix.
    R : np.ndarray
        Input weight matrix.
    V : np.ndarray
        Process noise covariance matrix.
    W : np.ndarray
        Measurement noise covariance matrix.

    Returns
    -------
    Ab : np.ndarray
        Balanced state matrix A.
    Bb : np.ndarray
        Balanced state matrix B.
    Cb : np.ndarray
        Balanced state matrix C.
    R : np.ndarray
        LQG-balanced transformation.
    Mu : np.ndarray
        Balanced CARE, FARE solutions.
    Qb : np.ndarray
        Balanced weight matrix.
    Vb : np.ndarray
        Balanced process noise covariance matrix.
    Kpb : np.ndarray
        Balanced gains.
    Keb : np.ndarray
        Balanced gains.
    """

    V1 = V
    R1 = R
    n1, n2 = A.shape
    
    # solution of CARE
    Kp, Sc, ec = control.lqr(A, B, Q, R)

    # solution of FARE
    Ke, Se, ee = control.lqe(A, np.eye(n1, dtype=complex), C, V, W)

    Uc, Ssc, Vc = svd(Sc)
    Pc = np.sqrt(Ssc) @ Vc.T
    
    Ue, Sse, Ve = svd(Se)
    Pe = Ue @ np.sqrt(Sse)
    
    H = Pc @ Pe
    
    V, Mu, U = svd(H)  # SVD of H
    mu = np.sqrt(Mu)
    R = Pe @ U @ inv(mu)  # transformation R
    
    Rinv = inv(mu) @ V.T @ Pc  # inverse of R
    Ab = Rinv @ A @ R
    Bb = Rinv @ B
    Cb = C @ R  # LQG balanced representation
    
    Qb = R.T @ Q @ R  # balanced weight matrix
    Vb = Rinv @ V1 @ Rinv.T  # balanced process noise cov. matrix

    # balanced gains
    Kpb, Scb, ecb = control.lqr(Ab, Bb, Qb, R1)
    Keb, Seb, eeb = control.lqe(Ab, np.eye(n1, dtype=complex), Cb, Vb, W)
    
    return Ab, Bb, Cb, Mu, Kpb, Keb, Qb, Vb, R


def bal_H_inf(A: np.ndarray, B1: np.ndarray, B2: np.ndarray, C1: np.ndarray,
              C2: np.ndarray, ro: float) -> \
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray]:
    """
    This function finds the H_inf-balanced representation
    (Ab, Bb1, Bb2, Cb1, Cb2) so that HCARE (Sc) and HFARE (Se) solutions are
    equal and diagonal:
    Sc = Se = Mu_inf

    Parameters
    ----------
    A : np.ndarray
        State matrix A.
    B1 : np.ndarray
        State matrix B1.
    B2 : np.ndarray
        State matrix B2.
    C1 : np.ndarray
        State matrix C1.
    C2 : np.ndarray
        State matrix C2.
    ro : float
        Parameter in HCARE and HFARE.

    Returns
    -------
    Ab : np.ndarray
        H_inf balanced state matrix A.
    Bb1 : np.ndarray
        H_inf balanced state matrix B1.
    Bb2 : np.ndarray
        H_inf balanced state matrix B2.
    Cb1 : np.ndarray
        H_inf balanced state matrix C1.
    Cb2 : np.ndarray
        H_inf balanced state matrix C2.
    R : np.ndarray
        H_inf balanced transformation.
    Mu : np.ndarray
        H_inf balanced HCARE, HFARE solutions.
    """
    n1, n2 = A.shape

    # Calculation of the matrices for HCARE
    Qc = C1.T @ C1
    gi = 1 / (ro * ro)
    Rc = B2 @ B2.T - gi * B1 @ B1.T

    # HCARE solution
    Sc = solve_continuous_are(A, np.eye(n1), Qc, inv(Rc))

    # Calculation of the matrices for HFARE
    Qe = B1 @ B1.T
    Re = C2.T @ C2 - gi * C1.T @ C1

    # HFARE solution
    Se = solve_continuous_are(A.T, np.eye(n1), Qe, inv(Re))

    # Check if solutions are real
    if (norm(np.imag(Se)) > 1e-6 or norm(np.imag(Sc)) > 1e-6):
        print('nonpositive solution')

    # Calculation of the transformation matrix
    Uc, Ssc, Vc = svd(Sc)
    Pc = np.sqrt(Ssc) @ Vc

    Ue, Sse, Ve = svd(Se)
    Pe = Ue @ np.sqrt(Sse)

    N = Pc @ Pe

    V, Mu_inf, U = svd(N)  # TODO: why are these outputs flipped?
    mu_inf = np.sqrt(Mu_inf)

    R = Pe @ U @ inv(mu_inf)
    Rinv = inv(mu_inf) @ V.T @ Pc

    # H_inf balanced representation
    Ab = Rinv @ A @ R
    Bb1 = Rinv @ B1
    Bb2 = Rinv @ B2
    Cb1 = C1 @ R
    Cb2 = C2 @ R

    return Ab, Bb1, Bb2, Cb1, Cb2, Mu_inf, R

