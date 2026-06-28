import numpy as np
from matplotlib import pyplot as plt
import bilby
from scipy.interpolate import CubicSpline
from scipy.special import erf
import shutil

import sys
sys.path.insert(0,'../src')
from util import *
from stochastic import *
from postprocessing import *
from pe import *
from constants import *

if __name__=='__main__':
    ####################
    # Inputs
    ####################

    # PSD ~ 1e-42 strain^2 / Hz
    # PSD = 2 sigma**2 * deltaT = 2 * sigma**2 / Fs
    # = sigma**2 / 512
    # sigma = sqrt(1e-42 * 512)

    sigma             = np.sqrt(512 * 1e-42) # strain
    sigma_inj         = sigma/10.
    NSegments         = 50
    Fs                = 1024. # Hz
    segmentDuration   = 64.0 # s
    t0                = 0 # s
    TAvg              = 4.0 # s

    ####################
    # Computed quantities
    ####################

    NSamplesPerSegment=int(segmentDuration*Fs) 
    deltaT=1/Fs
    fNyquist=1/(2*deltaT)
    deltaF=1/segmentDuration
    deltaFStoch=1/TAvg
    NAvgs = 2 * int(segmentDuration / TAvg) - 1
    jobDuration = NSegments * segmentDuration

    # Theoretical PSD and sigma for white GWB and white noise

    Nfreqs = int((Fs/2) / deltaFStoch)
    alpha=3 # white signal
    fref=25

    w1w2bar, w1w2squaredbar,_,_ = window_factors(NSamplesPerSegment)
    PSD_theor = 2*(sigma**2)*deltaT
    Pgw_theor = 2*(sigma_inj**2)*deltaT

    H_theor = (3*H0**2)/(10*np.pi**2*fref**3) # don't need *(P1.freqs/fref)**(alpha-3) for alpha=3
    Y_theor = Pgw_theor/H_theor

    var_theor = 1./((segmentDuration)*2*Nfreqs*deltaFStoch * H_theor**2/(PSD_theor**2))
    var_theor = w1w2squaredbar / w1w2bar**2 * var_theor
    sigma_theor = np.sqrt(var_theor)

    print('sigma_theor =', sigma_theor)
    print('Y_theor =', Y_theor)

    # simulate the time series data
    noise1=sigma*np.random.randn(int(NSamplesPerSegment*NSegments))
    noise2=sigma*np.random.randn(int(NSamplesPerSegment*NSegments))
    inj=sigma_inj*np.random.randn(int(NSamplesPerSegment*NSegments))

    times=np.arange(t0,t0+NSegments*segmentDuration,1/Fs)

    d1=TimeSeries(times,noise1 + inj)
    d2=TimeSeries(times,noise2 + inj)

    # run stochastic pipeline
    alpha=0
    fref=25
    Ys,sigs,Y_fs,var_fs,segmentStartTimes,freqs=stochastic(d1,d2,segmentDuration,deltaFStoch,fref=fref,alpha=alpha)

    # combine spectra over times

    Y_f,var_f=postprocessing_spectra(Y_fs,var_fs,jobDuration,segmentDuration,
                                 deltaFStoch,deltaT)

    # A few simple setup steps
    label = 'GWB_powerlaw'
    outdir = 'outdir'

    cleanup_dir(outdir)

    Amin,Amax,alpha_min,alpha_max=1e-10,1e-3,-5,5
    

    fref=25

    likelihood = BasicPowerLawGWBLikelihood(Y_f[1:],var_f[1:],freqs[1:],fref)
    priors = dict(A=bilby.core.prior.Uniform(Amin,Amax, 'A'),
              alpha=bilby.core.prior.Uniform(alpha_min,alpha_max, 'alpha'))

    # And run sampler
    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
        walks=10, outdir=outdir, label=label,maxmcmc=10000)
    fig = result.plot_corner()
    fig.savefig('results/bilby_corner.png')
    plt.close(fig)

    ##plt.savefig('results/bilby_corner.png')
    ##plt.close()


    # plot 1d A posterior
    A=result.samples[:,0]
    alpha=result.samples[:,1]

    plt.hist(A,bins=30,histtype='step',color='blue')
    plt.axvline(np.percentile(A,5),color='blue',linestyle='--')
    plt.axvline(np.percentile(A,95),color='blue',linestyle='--')

    plt.axvline(Y_theor,color='red')

    plt.title('A')
    plt.xlabel('')
    plt.savefig('results/A.png')
    plt.close()

    # plot 1d alpha posterior
    plt.hist(alpha,bins=30,histtype='step',color='blue')
    plt.axvline(np.percentile(alpha,5),color='blue',linestyle='--')
    plt.axvline(np.percentile(alpha,95),color='blue',linestyle='--')

    plt.axvline(3,color='red')

    plt.title('alpha')
    plt.xlabel('')
    plt.savefig('results/alpha.png')
    plt.close()
