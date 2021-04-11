from pathlib import Path
import numpy as np
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import animation


def loadIR(sessionPath):
    """Load impulse response (IR) data

    Parameters
    ------
    sessionPath: Path to IR folder

    Returns
    ------
    pos_mic: Microphone positions of shape (numMic, 3)
    pos_src: Source positions of shape (numSrc, 3)
    fullIR: IR data of shape (numSrc, numMic, irLen)
    """
    pos_mic = np.load(sessionPath.joinpath("pos_mic.npy"))
    pos_src = np.load(sessionPath.joinpath("pos_src.npy"))

    numMic = pos_mic.shape[0]
    
    allIR = []
    irIndices = []
    for f in sessionPath.iterdir():
        if not f.is_dir():
            if f.stem.startswith("ir_"):
                allIR.append(np.load(f))
                irIndices.append(int(f.stem.split("_")[-1]))

    assert(len(allIR) == numMic)
    numSrc = allIR[0].shape[0]
    irLen = allIR[0].shape[-1]
    fullIR = np.zeros((numSrc, numMic, irLen))
    for i, ir in enumerate(allIR):
        assert(ir.shape[0] == numSrc)
        assert(ir.shape[-1] == irLen)
        fullIR[:, irIndices[i], :] = ir

    return pos_mic, pos_src, fullIR


def sortIR(pos, ir, numXY, posX=None, posY=None):
    """Sort IR data into 2D rectangular shape
    """
    if (posX is None) or (posY is None):
        posX = np.unique(pos[:,0].round(4))
        posY = np.unique(pos[:,1].round(4))
    sortIdx = np.zeros((numXY[0], numXY[1]), dtype=int)
    
    for i in range(numXY[1]):
        xIdx = np.where(np.isclose(pos[:, 1], posY[i]))[0]
        sorter = np.argsort(pos[xIdx, 0])
        xIdxSort = xIdx[sorter]
        sortIdx[:,i] = xIdxSort

    sortPos = pos[sortIdx, :]
    sortIR = ir[:, sortIdx, :]
    return sortPos, sortIR, sortIdx


def sortIR3(pos, ir, numXYZ, posX=None, posY=None, posZ=None):
    """Sort IR data into 3D cuboid shape
    """
    if (posX is None) & (posY is None) & (posZ is None):
        posX = np.unique(pos[:, 0].round(4))
        posY = np.unique(pos[:, 1].round(4))
        posZ = np.unique(pos[:, 2].round(4))
    sortIdx = np.zeros((numXYZ[0], numXYZ[1], numXYZ[2]), dtype=int)
    
    for i in range(numXYZ[2]):
        for j in range(numXYZ[1]):
            xIdx = np.where(np.isclose(pos[:, 1], posY[j]) & np.isclose(pos[:, 2], posZ[i]))[0]
            sorter = np.argsort(pos[xIdx, 0])
            xIdxSort = xIdx[sorter]
            sortIdx[:, j, i] = xIdxSort

    sortPos = pos[sortIdx, :]
    sortIR = ir[:, sortIdx, :]
    return sortPos, sortIR, sortIdx


def extract_plane(pos, ir, z):
    """Extract IR data on the plane at z
    """
    z_list = pos[:, 2]
    pos_z_idx = np.where(z_list == z)[0].tolist()

    pos_z = pos[pos_z_idx, :]
    ir_z = ir[:, pos_z_idx, :]

    return pos_z, ir_z


def reverbParams(ir, samplerate):
    """Compute reverberation parameters
    Returns
    ------
    t60: Reverberation time RT60
    energy: Energy decay curve
    line: Regression line of energy decay curve
    """
    t = np.arange(ir.shape[0]) / samplerate
    energy = 10.0 * np.log10(np.cumsum(ir[::-1]**2)[::-1]/np.sum(ir**2))

    # Linear regression parameters for computing RT60
    init_db = -5
    end_db = -25
    factor = 3.0

    energy_init = energy[np.abs(energy - init_db).argmin()]
    energy_end = energy[np.abs(energy - end_db).argmin()]
    init_sample = np.where(energy == energy_init)[0][0]
    end_sample = np.where(energy == energy_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / samplerate
    y = energy[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]
    line = slope * t + intercept

    db_regress_init = (init_db - intercept) / slope
    db_regress_end = (end_db - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)

    return t60, energy, line


def irPlots(ir, samplerate):
    """Plot impulse response
    """
    t = np.arange(ir.shape[0]) / samplerate

    rt60, energy_curve, energy_line = reverbParams(ir, samplerate)
    print("RT60 (ms): ", '{:.1f}'.format(rt60*1000))

    f_spec, t_spec, spec = signal.spectrogram(ir, samplerate, nperseg=512)

    # IR
    plt.plot(t, ir)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    # Energy decay curve
    # plt.plot(t, energy_curve)
    # plt.plot(t, energy_line, linestyle="--")
    # plt.ylim(-70, 5)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Energy (dB)')
    # plt.show()

    # Spectrogram
    # color = plt.pcolormesh(t_spec, f_spec, 20*np.log10(spec), vmin=-250, shading='auto')
    # cbar=plt.colorbar(color)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # cbar.set_label('Power (dB)')
    # plt.show()


def plotWave(x, y, ir, tIdx=None):
    """Plot instantaneous pressure distribution
    """
    if tIdx is None:
        tIdx, _ = findPeak(ir, 0)
        print("Time (sample):", tIdx)

    xx, yy = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    ax = plt.axes()
    color = plt.pcolor(xx, yy, ir[:, :, tIdx], cmap='RdBu', shading='auto')
    ax.set_aspect('equal')
    cbar = plt.colorbar(color)
    cbar.set_label('Amplitude')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()


def plotWaveFronts(x, ir, samplerate, xy='x'):
    """Plot impulse responses along the line at x
    """
    tIdxMin, tIdxMax = findPeak(ir)
    t = np.arange(tIdxMin, tIdxMax)/samplerate
    if xy == 'x':
        ir_plt = np.squeeze(ir[:, 0, tIdxMin:tIdxMax])
    elif xy == 'y':
        ir_plt = np.squeeze(ir[0, :, tIdxMin:tIdxMax])
    else:
        raise ValueError()

    xx, yy = np.meshgrid(t, x)
    fig, ax = plt.subplots()
    ax = plt.axes()
    color = plt.pcolor(xx, yy, ir_plt, cmap='RdBu', shading='auto')
    cbar = plt.colorbar(color)
    cbar.set_label('Amplitude')
    plt.xlabel('Time (s)')
    if xy == 'x':
        plt.ylabel('x (m)')
    elif xy == 'y':
        plt.ylabel('y (m)')
    plt.show()


def findPeak(ir, preBuffer=100, tailBuffer=100):
    """Find time sample of peak amplitude
    """
    peakIdx = np.argmax(np.abs(ir), axis=-1)
    minPeakIdx = np.min(peakIdx)
    maxPeakIdx = np.max(peakIdx)
    return minPeakIdx-preBuffer, maxPeakIdx+tailBuffer


def drawGeometry(posSrc, posMic):
    """Plot geometry of sources and microphones 
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(posMic[:,0], posMic[:,1], posMic[:,2], marker='.')
    ax.scatter3D(posSrc[:,0], posSrc[:,1], posSrc[:,2], marker='*')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    plt.show()


def movWave(sessionPath, x, y, ir, samplerate, start=None, end=None, downSampling=None):
    """Generate movie of pressure field
    """
    if (start is None) or (end is None):
        start, end = findPeak(ir)

    if downSampling is not None:
        ir = signal.resample_poly(ir, up=1, down=downSampling, axis=-1)
        samplerate = samplerate // downSampling
    
    maxVal = np.max(np.abs(ir))

    plt.rcParams["font.size"] = 14

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(ir[..., start], vmin=-maxVal, vmax=maxVal, cmap='RdBu', origin='lower', interpolation='none')
    cbar = fig.colorbar(cax)
    ax.set_xticks(np.arange(0, x.shape[0], 4))
    ax.set_xticklabels(x[0::4])
    ax.set_yticks(np.arange(0, y.shape[0], 4))
    ax.set_yticklabels(y[0::4])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    def animate(i):
        cax.set_array(ir[..., start+i])
        currentTime = (i+start) / samplerate
        ax.set_title("Sample: "+ str(i)+ ", Time: "+ "{:.3f}".format(currentTime), fontsize = 14)
        return cax,
        
    anim = animation.FuncAnimation(fig, animate, interval=15, frames=end-start-1, blit=True)

    anim.save(sessionPath.joinpath("wave_mov.mp4"), writer='ffmpeg', fps=15, bitrate=1800)

    plt.draw()
    plt.show()


if __name__ == "__main__":
    sessionName = "S32-M441_npy"  # "S1-M3969_npy"
    sessionPath = Path(__file__).parent.joinpath(sessionName)

    # Load files
    posMic, posSrc, ir = loadIR(sessionPath)

    # Sampling rate
    samplerate = 48000
    srcIdx = 0
    micIdx = 0
    print("Source position (m): ", posSrc[srcIdx, :])
    print("Mic position (m): ", posMic[micIdx, :])

    # Geometry
    drawGeometry(posSrc, posMic)

    # IR plots
    ir_plt = ir[srcIdx, micIdx, :]
    irPlots(ir_plt, samplerate)
    
    # Extract plane
    z = 0.0
    posMic_z, ir_z = extract_plane(posMic, ir, z)
    posMicX = np.unique(posMic_z[:, 0].round(4))
    posMicY = np.unique(posMic_z[:, 1].round(4))
    numXY = (posMicX.shape[0], posMicX.shape[0])
    posMicXY, irXY, _ = sortIR(posMic_z, ir_z, numXY, posMicX, posMicY)
    
    # Lowpass filter
    maxFreq = 600
    h = signal.firwin(numtaps=64, cutoff=maxFreq, fs=samplerate)
    irXY_lp = signal.filtfilt(h, 1, irXY[srcIdx, :, :, :], axis=-1)

    # Wave image
    plotWave(posMicX, posMicY, irXY_lp)
    # plotWaveFronts(posMicX, irXY_lp, samplerate)

    # Wave movie
    # movWave(sessionPath, posMicX, posMicY, irXY_lp, samplerate)
    
