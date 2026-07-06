from __future__ import division
import numpy as np
import scipy as scipy
from scipy.stats import norm
import matplotlib.pylab as plt
import matplotlib.mlab as mlab

def combinesignalnoise(signalfile, noisefile, A, fileprefix):

    '''
    combine signal + noise from signal and noise data files, writing combined data to fileprefix.txt
    '''
    
    ts = np.loadtxt(signalfile)
    t = ts[:,0]
    s = A*ts[:,1]
    
    ts = np.loadtxt(noisefile)
    t = ts[:,0]
    n = ts[:,1]

    d = s + n
    
    savetimeseries(t, d, fileprefix)
    
    return t, d, s, n


def combinesignalsignal(signalfile1, signalfile2, A1, A2, fileprefix):

    '''
    combine signal + signal from two signal data files, writing combined data to fileprefix.txt
    '''
    
    ts = np.loadtxt(signalfile1)
    t = ts[:,0]
    s1 = A1 * ts[:,1]
    
    ts = np.loadtxt(signalfile2)
    t = ts[:,0]
    s2 = A2 * ts[:,1]

    d = s1 + s2
    
    savetimeseries(t, d, fileprefix)
    
    return t, d, s1, s2


def plotpowerspectrum(t, y, t1, t2, Fs, fileprefix):

    '''
    plot powerspectrum of time-series data between t1 and t2 and save plot to .png file
    '''
    
    filename = fileprefix + '_powerspectrum.png'
        
    # find indices for tlow, thigh
    n1 = np.where(t>=t1)[0]
    n2 = np.where(t>=t2)[0]
 
    # calculate welch estimate of power spectrum
    N = n2[0]-n1[0]+1;
    seglength = int(N/8.)
    f, P = scipy.signal.welch(y[n1[0]:n2[0]], fs=Fs, window='hann', nperseg=seglength)
    
    # plot power spectrum
    plt.figure()
    plt.rc('text', usetex=True)
    plt.tick_params(labelsize=20)
    plt.loglog(f, P, linewidth=2)
    plt.xlabel('frequency (Hz)', size=22)
    plt.ylabel('power spectral density (1/Hz)', size=22)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    
    return


def plotsignalnoisetimeseries(t, s, d, t1, t2, fileprefix):

    '''
    plot signal and data between t1 and t2 and save plot to .png file
    '''
    
    filename = fileprefix + '.png'
        
    # find indices for tlow, thigh
    n1 = np.where(t>=t1)[0]
    n2 = np.where(t>=t2)[0]
    
    plt.figure()
    plt.rc('text', usetex=True)
    plt.tick_params(labelsize=20)
    plt.plot(t[n1[0]:n2[0]], d[n1[0]:n2[0]], color='k')
    plt.plot(t[n1[0]:n2[0]], s[n1[0]:n2[0]], color='r')
        
    # set symmetric ylimits if both positive and negative
    axes = plt.gca()
    y1, y2 = axes.get_ylim()
    if y1<0. and y2>0.:
        ymin = -max(np.abs(y1),np.abs(y2))
        ymax =  max(np.abs(y1),np.abs(y2))
        axes.set_ylim([ymin, ymax])
    
    plt.xlabel('time (s)', size=22)
    plt.ylabel('data', size=22)
    plt.legend(['data', 'signal'])
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    
    return


def plotsignalsignalnoisetimeseries(t, s1, s2, d, t1, t2, fileprefix):

    '''
    plot two signal components and data between t1 and t2 and save plot to .png file
    '''
    
    filename = fileprefix + '.png'
        
    # find indices for tlow, thigh
    n1 = np.where(t>=t1)[0]
    n2 = np.where(t>=t2)[0]
    
    plt.figure()
    plt.rc('text', usetex=True)
    plt.tick_params(labelsize=20)
    plt.plot(t[n1[0]:n2[0]], d[n1[0]:n2[0]], color='k')
    plt.plot(t[n1[0]:n2[0]], s1[n1[0]:n2[0]], color='r')
    plt.plot(t[n1[0]:n2[0]], s2[n1[0]:n2[0]], color='g')
        
    # set symmetric ylimits if both positive and negative
    axes = plt.gca()
    y1, y2 = axes.get_ylim()
    if y1<0. and y2>0.:
        ymin = -max(np.abs(y1),np.abs(y2))
        ymax =  max(np.abs(y1),np.abs(y2))
        axes.set_ylim([ymin, ymax])
    
    plt.xlabel('time (s)', size=22)
    plt.ylabel('data', size=22)
    plt.legend(['data', 'signal 1', 'signal 2'])
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    
    return


def plotsignalsignaltimeseries(t, s1, s2, t1, t2, fileprefix):

    '''
    plot two signal components between t1 and t2 and save plot to .png file
    '''
    
    filename = fileprefix + '.png'
        
    # find indices for tlow, thigh
    n1 = np.where(t>=t1)[0]
    n2 = np.where(t>=t2)[0]
    
    plt.figure()
    plt.rc('text', usetex=True)
    plt.tick_params(labelsize=20)
    plt.plot(t[n1[0]:n2[0]], s1[n1[0]:n2[0]], color='r')
    plt.plot(t[n1[0]:n2[0]], s2[n1[0]:n2[0]], color='g')
        
    # set symmetric ylimits if both positive and negative
    axes = plt.gca()
    y1, y2 = axes.get_ylim()
    if y1<0. and y2>0.:
        ymin = -max(np.abs(y1),np.abs(y2))
        ymax =  max(np.abs(y1),np.abs(y2))
        axes.set_ylim([ymin, ymax])
    
    plt.xlabel('time (s)', size=22)
    plt.ylabel('data', size=22)
    plt.legend(['signal 1', 'signal 2'])
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    
    return


def plottimeseries(t, y, t1, t2, fileprefix):

    '''
    plot time-series data between t1 and t2 and save plot to .png file
    '''
    
    filename = fileprefix + '.png'
        
    # find indices for tlow, thigh
    n1 = np.where(t>=t1)[0]
    n2 = np.where(t>=t2)[0]
    
    plt.figure()
    plt.rc('text', usetex=True)
    plt.tick_params(labelsize=20)
    plt.plot(t[n1[0]:n2[0]], y[n1[0]:n2[0]])
        
    # set symmetric ylimits if both positive and negative
    axes = plt.gca()
    y1, y2 = axes.get_ylim()
    if y1<0. and y2>0.:
        ymin = -max(np.abs(y1),np.abs(y2))
        ymax =  max(np.abs(y1),np.abs(y2))
        axes.set_ylim([ymin, ymax])
    
    plt.xlabel('time (s)', size=22)
    plt.ylabel('data', size=22)
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    
    return


def savetimeseries(t, y, fileprefix):
    
    '''
    save timeseries data to file
    '''

    filename = fileprefix + '.txt'
    N = len(t)
    ts = np.zeros([N,2])
    ts[:,0] = t
    ts[:,1] = y
    np.savetxt(filename, ts)

    return
