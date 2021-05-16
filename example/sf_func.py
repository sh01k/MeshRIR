import numpy as np
import scipy.special as special
import scipy.spatial.distance as distfuncs


def cart2sph(x, y, z):
    """Conversion from Cartesian to spherical coordinates

    Parameters
    ------
    x, y, z : Position in Cartesian coordinates

    Returns
    ------
    phi, theta, r: Azimuth angle, zenith angle, distance
    """
    r_xy = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(r_xy, z)
    r = np.sqrt(x**2 + y**2 + z**2)
    return phi, theta, r


def sph2cart(phi, theta, r):
    """Conversion from spherical to Cartesian coordinates

    Parameters
    ------
    phi, theta, r: Azimuth angle, zenith angle, distance

    Returns
    ------
    x, y, z : Position in Cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def sph_harm(m, n, phi, theta):
    """Spherical harmonic function
    m, n: degrees and orders
    phi, theta: azimuth and zenith angles
    """
    return special.sph_harm(m, n, phi, theta)


def sph_harm_nmvec(order, rep=None):
    """Vectors of spherical harmonic orders and degrees
    Returns (order+1)**2 size vectors of n and m
    n = [0, 1, 1, 1, ..., order, ..., order]^T
    m = [0, -1, 0, 1, ..., -order, ..., order]^T

    Parameters
    ------
    order: Maximum order
    rep: Same vectors are copied as [n, .., n] and [m, ..., m]

    Returns
    ------
    n, m: Vectors of orders and degrees
    """
    n = np.array([0])
    m = np.array([0])
    for nn in np.arange(1, order+1):
        nn_vec = np.tile([nn], 2*nn+1)
        n = np.append(n, nn_vec)
        mm = np.arange(-nn, nn+1)
        m = np.append(m, mm)
    if rep is not None:
        n = np.tile(n[:, None], (1, rep))
        m = np.tile(m[:, None], (1, rep))
    return n, m


def spherical_hn(n, k, z):
    """nth-order sphericah Henkel function of kth kind
    Returns h_n^(k)(z)
    """
    if k == 1:
        return special.spherical_jn(n, z) + 1j * special.spherical_yn(n, z)
    elif k == 2:
        return special.spherical_jn(n, z) - 1j * special.spherical_yn(n, z)
    else:
        raise ValueError()


def sf_int_basis3d(n, m, x, y, z, k):
    """Spherical wavefunction for interior sound field in 3D
    
    Parameters
    ------
    n, m: orders and degrees
    x, y, z: Position in Cartesian coordinates
    k: Wavenumber

    Returns
    ------
    sqrt(4pi) j_n(kr) Y_n^m(phi,theta)
    (Normalized so that 0th order coefficient corresponds to pressure)
    """
    phi, theta, r = cart2sph(x, y, z)
    J = special.spherical_jn(n, k * r)
    Y = sph_harm(m, n, phi, theta)
    f = np.sqrt(4*np.pi) * J * Y
    return f


def gauntcoef(l1, m1, l2, m2, l3):
    """Gaunt coefficients
    """
    m3 = -m1 - m2
    l = int((l1 + l2 + l3) / 2)

    t1 = l2 - m1 - l3
    t2 = l1 + m2 - l3
    t3 = l1 + l2 - l3
    t4 = l1 - m1
    t5 = l2 + m2

    tmin = max([0, max([t1, t2])])
    tmax = min([t3, min([t4, t5])])

    t = np.arange(tmin, tmax+1)
    gl_tbl = np.array(special.gammaln(np.arange(1, l1+l2+l3+3)))
    G = np.sum( (-1.)**t * np.exp( -np.sum( gl_tbl[np.array([t, t-t1, t-t2, t3-t, t4-t, t5-t])] )  \
                                  +np.sum( gl_tbl[np.array([l1+l2-l3, l1-l2+l3, -l1+l2+l3, l])] ) \
                                  -np.sum( gl_tbl[np.array([l1+l2+l3+1, l-l1, l-l2, l-l3])] ) \
                                  +np.sum( gl_tbl[np.array([l1+m1, l1-m1, l2+m2, l2-m2, l3+m3, l3-m3])] ) * 0.5 ) ) \
        * (-1.)**( l + l1 - l2 - m3) * np.sqrt( (2*l1+1) * (2*l2+1) * (2*l3+1) / (4*np.pi) )
    return G


def trjmat3d(order1, order2, x, y, z, k):
    """Translation operator in 3D
    """
    if np.all([x, y, z] == 0):
        T = np.eye((order1+1)**2, (order2+1)**2)
        return T
    else:
        order = order1 + order2
        n, m = sph_harm_nmvec(order)
        P = sf_int_basis3d(n, m, x, y, z, k)
        T = np.zeros(((order1+1)**2, (order2+1)**2), dtype=complex)

        icol = 0
        for n in np.arange(0, order2+1):
            for m in np.arange(-n, n+1):
                irow = 0
                for nu in np.arange(0, order1+1):
                    for mu in np.arange(-nu, nu+1):
                        l = np.arange((n+nu), max( [np.abs(n-nu), np.abs(m-mu)] )-1, -2)
                        G = np.zeros(l.shape)
                        for ll in np.arange(0, l.shape[0]):
                            G[ll] = gauntcoef(n, m, nu, -mu, l[ll])
                        T[irow, icol] = np.sqrt(4.*np.pi) * 1j**(nu-n) * (-1.)**m * np.sum( 1j**(l) * P[l**2 + l - (mu-m)] * G )
                        irow = irow + 1
                icol = icol+1
        return T


def planewave(amp, phi, theta, x, y, z, k):
    """Planewave
    """
    kx, ky, kz = sph2cart(phi, theta, k)
    p = amp * np.exp(-1j * (kx * x + ky * y + kz * z))
    return p


def planewave_mode(order, amp, phi, theta, x, y, z, k):
    """Expansion coefficients of planewave by spherical wavefunctions
    """
    kx, ky, kz = sph2cart(phi, theta, k)
    A = amp * np.exp(-1j * (kx*x + ky*y + kz*z))
    n, m = sph_harm_nmvec(order, 1)
    coef = A * np.sqrt(4 * np.pi) * (-1j)**n * sph_harm(m, n, phi, theta).conj()
    return coef


def sphericalwave(amp, x_s, y_s, z_s, x, y, z, k):
    """Point source (3D free-field Green's function)
    """
    r = np.sqrt((x-x_s)**2 + (y-y_s)**2 + (z-z_s)**2)
    p = amp * np.exp(- 1j * k * r) / (4 * np.pi * r)
    return p


def sphericalwave_mode(order, amp, x_s, y_s, z_s, x, y, z, k):
    """Expansion coefficients of point source by spherical wavefunctions
    """
    phi_s, theta_s, r_s = cart2sph(x_s-x, y_s-y, z_s-z)
    n, m = sph_harm_nmvec(order, 1)
    if np.isscalar(phi_s) is False:
        numPos = phi_s.shape[0]
        numOrd = n.shape[0]
        n = np.tile(n, (1,numPos))
        m = np.tile(m, (1,numPos))
        amp = np.tile(amp.T, (numOrd, 1))
        phi_s = np.tile(phi_s, (numOrd, 1))
        theta_s = np.tile(theta_s, (numOrd, 1))
        r_s = np.tile(r_s, (numOrd, 1))
    coef = - amp * 1j * k / np.sqrt(4 * np.pi) * spherical_hn(n, 2, k*r_s) * sph_harm(m, n, phi_s, theta_s).conj()
    return coef


def coefEstOprGen(posEst, orderEst, posMic, orderMic, coefMic, k):
    """Generate operator to estimate expansion coefficients of spherical wavefunctions from measurement vectors
    - N. Ueno, S. Koyama, and H. Saruwatari, “Sound Field Recording Using Distributed Microphones Based on 
      Harmonic Analysis of Infinite Order,” IEEE SPL, DOI: 10.1109/LSP.2017.2775242, 2018.

    Parameters
    ------
    posEst: Position of expansion center for estimation
    orderEst: Maximum order for estimation
    poMic: Microphone positions
    orderMic: Maximum order of microphone directivity
    coefMic: Expansion coefficients of microphone directivity

    Returns
    ------
    Operator for estimation
    (Expansion coefficeints are estimated by multiplying with measurement vectors)
    """
    reg = 1e-3
    numMic = posMic.shape[0]
    if np.isscalar(k):
        numFreq = 1
        k = np.array([k])
    else:
        numFreq = k.shape[0]

    Xi = np.zeros((numFreq, (orderEst+1)**2, numMic), dtype=complex)
    Psi = np.zeros((numFreq, numMic, numMic), dtype=complex)
    for ff in np.arange(numFreq):
        print('Frequency: %d/%d' % (ff, numFreq))
        for j in np.arange(numMic):
            T = trjmat3d(orderEst, orderMic, posEst[0, 0]-posMic[j, 0], posEst[0, 1]-posMic[j, 1], posEst[0, 2]-posMic[j, 2], k[ff])
            Xi[ff, :, j] = T @ coefMic[:, j]
            Psi[ff, j, j] = coefMic[:, j].conj().T @ coefMic[:, j]
            for i in np.arange(j, numMic):
                T = trjmat3d(orderMic, orderMic, posMic[i, 0]-posMic[j, 0], posMic[i, 1]-posMic[j, 1], posMic[i, 2]-posMic[j, 2], k[ff])
                Psi[ff, i, j] = coefMic[:, i].conj().T @ T @ coefMic[:, j]
                Psi[ff, j, i] = Psi[ff, i, j].conj()
    Psi_inv = np.linalg.inv(Psi + reg * np.eye(numMic, numMic)[None, :, :])
    coefEstOpr = Xi @ Psi_inv

    return coefEstOpr


def kiFilterGen(k, posMic, posEst, filterLen=None, smplShift=None):
    """Kernel interpolation filter for estimating pressure distribution from measurements
    - N. Ueno, S. Koyama, and H. Saruwatari, “Kernel Ridge Regression With Constraint of Helmholtz Equation 
      for Sound Field Interpolation,” Proc. IWAENC, DOI: 10.1109/IWAENC.2018.8521334, 2018.
    - N. Ueno, S. Koyama, and H. Saruwatari, “Sound Field Recording Using Distributed Microphones Based on 
      Harmonic Analysis of Infinite Order,” IEEE SPL, DOI: 10.1109/LSP.2017.2775242, 2018.
    """
    numMic = posMic.shape[0]
    numEst = posEst.shape[0]
    numFreq = k.shape[0]
    fftlen = numFreq*2
    reg = 1e-1

    if filterLen is None:
        filterLen = numFreq+1
    if smplShift is None:
        smplShift = numFreq/2

    k = k[:, None, None]
    distMat = distfuncs.cdist(posMic, posMic)[None, :, :]
    K = special.spherical_jn(0, k * distMat)
    Kinv = np.linalg.inv(K + reg * np.eye(numMic)[None, :, :])
    distVec = np.transpose(distfuncs.cdist(posEst, posMic), (1, 0))[None, :, :]
    kappa = special.spherical_jn(0, k * distVec)
    kiTF = np.transpose(kappa, (0, 2, 1)) @ Kinv
    kiTF = np.concatenate((np.zeros((1, numEst, numMic)), kiTF, kiTF[int(fftlen/2)-2::-1, :, :].conj()))
    kiFilter = np.fft.ifft(kiTF, n=fftlen, axis=0).real
    kiFilter = np.concatenate((kiFilter[fftlen-smplShift:fftlen, :, :], kiFilter[:filterLen-smplShift, :, :]))

    return kiFilter


def kiFilterGenDir(k, posMic, posEst, angSrc, betaSrc, filterLen=None, smplShift=None):
    """Kernel interpolation filter with directional weighting for estimating pressure distribution from measurements
    - N. Ueno, S. Koyama, and H. Saruwatari, “Directionally Weighted Wave Field Estimation Exploiting Prior 
      Information on Source Direction,” IEEE Trans. SP, DOI: 10.1109/TSP.2021.3070228, 2021.
    """
    numMic = posMic.shape[0]
    numEst = posEst.shape[0]
    numFreq = k.shape[0]
    fftlen = numFreq*2
    reg = 1e-1

    if filterLen is None:
        filterLen = numFreq+1
    if smplShift is None:
        smplShift = int(numFreq/2)

    thetaSrc = angSrc[0]
    phiSrc = angSrc[1]

    k = k[:, None, None]
    rDiffMat = (np.tile(posMic[:, None, :], (1, numMic, 1)) - np.tile(posMic[None, :, :], (numMic, 1, 1)))[None, :, :, :]
    distMat = np.sqrt((1j*betaSrc*np.sin(thetaSrc)*np.cos(phiSrc) - k*rDiffMat[:, :, :, 0])**2 + (1j*betaSrc*np.sin(thetaSrc)*np.sin(angSrc) - k*rDiffMat[:, :, :, 1])**2 + (1j*betaSrc*np.cos(thetaSrc) - k*rDiffMat[:, :, :, 2])**2)
    K = special.spherical_jn(0, distMat)
    Kinv = np.linalg.inv(K + reg * np.eye(numMic)[None, :, :])
    rDiffVec = (np.tile(posEst[None, :, :], (numMic, 1, 1)) - np.tile(posMic[:, None, :], (1, numEst, 1)))[None, :, :, :]
    distVec = np.sqrt((1j*betaSrc*np.cos(angSrc) - k*rDiffVec[:, :, :, 0])**2 + (1j*betaSrc*np.sin(angSrc) - k*rDiffVec[:, :, :, 1])**2)
    kappa = special.spherical_jn(0, distVec)
    kiTF = np.transpose(kappa, (0, 2, 1)) @ Kinv
    kiTF = np.concatenate((np.zeros((1, numEst, numMic)), kiTF, kiTF[int(fftlen/2)-2::-1, :, :].conj()))
    kiFilter = np.fft.ifft(kiTF, n=fftlen, axis=0).real
    kiFilter = np.concatenate((kiFilter[fftlen-smplShift:fftlen, :, :], kiFilter[:filterLen-smplShift, :, :]))

    return kiFilter


def mcBlockRect(rectDims, rng=None):
    """Generate sampling points for Monte-Carlo integration inside rectangular region
    """
    if rng is None:
        rng = np.random.RandomState()
    totVol = rectDims[0] * rectDims[1]

    def pointGenerator(numSamples):
        x = rng.uniform(-rectDims[0] / 2, rectDims[0] / 2, numSamples)
        y = rng.uniform(-rectDims[1] / 2, rectDims[1] / 2, numSamples)
        points = np.stack((x, y))
        return points.T

    return pointGenerator, totVol


def wightBasis(k, n, m):
    """Integrand for weighted mode-matching
    """
    def integralFunc(r):
        funcVal = sf_int_basis3d(n[None,None,:,:], m[None,None,:,:], r[:,0,None,None,None], r[:,1,None,None,None], 0., k[None,:,None,None]) @ np.transpose( sf_int_basis3d(n[None,None,:,:], m[None,None,:,:], r[:,0,None,None,None], r[:,1,None,None,None], 0., k[None,:,None,None]).conj(), (0, 1, 3, 2))
        funcVal = np.transpose(funcVal, (1,2,3,0))
        return funcVal
    return integralFunc


def mcIntegrate(func, pointGenerator, totNumSamples, totalVolume, numPerIter=20):
    """Monte-Carlo integration
    """
    samplesPerIter = numPerIter
    numBlocks = int(np.ceil(totNumSamples / samplesPerIter))
    outDims = np.squeeze(func(pointGenerator(1)), axis=-1).shape
    integralVal = np.zeros(outDims)
    print(
        "Starting monte carlo integration \n",
        "Samples per block: ",
        numPerIter,
        "\nTotal samples: ",
        numBlocks * numPerIter,
    )

    for i in range(numBlocks):
        points = pointGenerator(samplesPerIter)
        fVals = func(points)

        newIntVal = (integralVal * i + np.mean(fVals, axis=-1)) / (i + 1)
        print("Block ", i)

        integralVal = newIntVal
    integralVal *= totalVolume
    print("Finished!!")
    return integralVal


def weightWMM(k, order, mcNumPoints, dimsEval):
    """Weighting matrix of weighted mode-matching method for sound field synthesis
    - N. Ueno, S. Koyama, and H. Saruwatari, “Three-Dimensional Sound Field Reproduction Based on 
      Weighted Mode-Matching Method,” IEEE/ACM Trans. ASLP, DOI: 10.1109/TASLP.2019.2934834, 2019.
    """
    n, m = sph_harm_nmvec(order, 1)
    mcPointGen, mcVolume = mcBlockRect([dimsEval[0], dimsEval[1]], np.random.RandomState(2))
    func = wightBasis(k, n, m)
    W = mcIntegrate(func, mcPointGen, mcNumPoints, mcVolume)
    return W


if __name__ == "__main__":
    from scipy import signal
    from pathlib import Path
    import matplotlib.pyplot as plt

    import sys
    sys.path.append(str(Path('__file__').resolve().parents[1]))
    import irutilities as irutil

    # Load IR data
    sessionName = "S32-M441_npy"
    sessionPath = Path('__file__').resolve().parents[1].joinpath(sessionName)
    posEval, posSrc, irAll = irutil.loadIR(sessionPath)

    numEval = posEval.shape[0]  # number of evaluation points
    numSrc = posSrc.shape[0]  # number of sources

    # Evaluation points in 2D
    posEvalX = np.unique(posEval[:,0].round(4))
    posEvalY = np.unique(posEval[:,1].round(4))
    numEvalXY = (posEvalX.shape[0], posEvalY.shape[0])

    # Sampling rate (original)
    samplerate_raw = 48000

    # Downsampling
    downSampling = 6
    irEval = signal.resample_poly(irAll, up=1, down=downSampling, axis=-1)
    samplerate = samplerate_raw // downSampling
    print('samplerate (Hz): ', samplerate)

    posEvalXY, _, idxEvalXY = irutil.sortIR(posEval[:,0:2], irEval, numEvalXY, posEvalX, posEvalY)

    # Trucation of IRs
    irLen = 4096  # Truncation length
    irEval = np.transpose(irEval, (2,1,0))
    irEval = irEval[0:irLen, :, :]
    print('ir length:', irLen)

    # Control points
    xGrid = np.arange(np.min(posEvalX)+0.05, np.max(posEvalX), 0.1)
    yGrid = np.arange(np.min(posEvalY)+0.05, np.max(posEvalY), 0.1)

    xIdx = []
    for ii in np.arange(xGrid.shape[0]):
        xIdx.append(np.where( np.isclose(posEval[:,0], xGrid[ii]) ))
    yIdx = []
    for ii in np.arange(yGrid.shape[0]):
        yIdx.append(np.where( np.isclose(posEval[:,1], yGrid[ii]) ))

    idxMic = np.intersect1d(xIdx, yIdx)
    numMic = idxMic.shape[0]
    posMic = posEval[idxMic,:]

    # IR at control points
    irMic = irEval[:,idxMic,:]

    # FFT parameters
    fftlen = 16384
    freq = np.arange(1,int(fftlen/2)+1)/fftlen*samplerate  # Frequency
    numFreq = freq.shape[0]  # Number of frequency bins
    c = 341.9  # Sound speed
    k = 2.0 * np.pi * freq / c  # Wavenumber

    # Transfer function matrix
    tfMic = np.fft.fft(irMic, n=fftlen, axis=0)
    G = tfMic[1:int(fftlen/2)+1,:,:]  # Transfer functions of positive frequencies

    # Setting for estimation of expansion coefficients
    posEst = np.zeros((1,3))  # Origin
    orderMic = 0  # Pressure microphone
    orderEst = 9  # Maximum order for estimation
    coefMic = np.ones((1, numMic), dtype=complex)  # Expansion coefficients of microphone directivity

    # Load estimation operator
    fileName = 'coefEstOp_o%d_m%d_f%d.npy' % (orderEst, numMic, numFreq)
    filePath = Path('__file__').parent.joinpath(fileName)
    coefEstOpr = np.load(filePath)
    # To regenerate (and save) a new estimation operator, uncomment the following lines 
    # coefEstOpr = coefEstOprGen(posEst, orderEst, posMic, orderMic, coefMic, k)
    # np.save(filePath, coefEstOpr)

    # Estimation of expantion coefficients of loudspeaker transfer functions
    coefEst = np.zeros((k.shape[0], (orderEst+1)**2, numSrc), dtype=complex)
    for ss in np.arange(numSrc):
        sigG = G[:, :, ss]
        coefEst[:, :, ss] = np.squeeze( coefEstOpr @ sigG[:, :, None] )

    # Filter parameters
    smplShift = 0
    filterLen = 4096

    # Lowpass filter
    maxFreq = 500
    h = signal.firwin(numtaps=64, cutoff=maxFreq, fs=samplerate)

    # Translation operator from origin to evaluation positions
    fileName = 'transOrg2Eval_o%d_e%d_f%d.npy' % (orderEst, numEval, numFreq)
    filePath = Path('__file__').parent.joinpath(fileName)
    T = np.load(filePath)

    # Estimation of pressure distribution using expantion coefficients
    ss = 0
    coefEst0 = coefEst[:, :, ss]
    # To regenerate (and save) translation operator, uncomment the following lines
    # coefEst0 = sphericalwave_mode(orderEst, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, k[None,:]).T
    # T = np.zeros((numFreq, 1, (orderEst+1)**2, numEval), dtype=complex)
    sfEst = np.zeros((numFreq, numEval), dtype=complex)
    for rr in np.arange(0, numEval):
        # print('Evaluation: %d/%d' % (rr, numEval))
        # for ff in np.arange(numFreq):
        #     T[ff,:,:,rr] = trjmat3d(0, orderEst, posEval[rr,0]-posEst[0,0], posEval[rr,1]-posEst[0,1], posEval[rr,2]-posEst[0,2], k[ff])
        sfEst[:,rr] = np.squeeze(T[:,:,:,rr] @ coefEst0[:,:,None])
    # np.save(filePath, T)

    sfEst = np.concatenate( (np.zeros((1,numEval)), sfEst, sfEst[int(fftlen/2)-2::-1,:].conj()) )
    sigEst = np.fft.ifft(sfEst, n=fftlen, axis=0).real
    sigEst = np.concatenate((sigEst[fftlen-smplShift:fftlen,:], sigEst[:filterLen-smplShift,:]))

    # Lowpass
    sigEval = irEval[:,:,ss]
    sigEvalLp = signal.filtfilt(h, 1, sigEval, axis=0)
    sigEvalLp_XY = sigEvalLp[:,idxEvalXY]
    sigEstLp = signal.filtfilt(h, 1, sigEst, axis=0)
    sigEstLp_XY = sigEstLp[:,idxEvalXY]

    # Plot signals
    fig, ax = plt.subplots()
    ax.plot(sigEvalLp[:,0])
    plt.xlabel('Sample')

    fig, ax = plt.subplots()
    ax.plot(sigEstLp[:,0])
    plt.xlabel('Sample')

    plt.show()

    # Draw pressure distribution
    tIdx = 980

    xx, yy = np.meshgrid(posEvalX, posEvalY)

    fig, ax = plt.subplots()
    ax = plt.axes()
    color = plt.pcolor(xx, yy, sigEvalLp_XY[tIdx,:,:], cmap='RdBu', shading='auto', vmin=-0.2, vmax=0.2)
    ax.set_aspect('equal')
    cbar=plt.colorbar(color)
    cbar.set_label('Amplitude')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    fig, ax = plt.subplots()
    ax = plt.axes()
    color = plt.pcolor(xx, yy, sigEstLp_XY[tIdx,:,:], cmap='RdBu', shading='auto', vmin=-0.2, vmax=0.2)
    ax.set_aspect('equal')
    cbar=plt.colorbar(color)
    cbar.set_label('Amplitude')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    plt.show()