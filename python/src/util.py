import numpy as np
from matplotlib import pyplot as plt
import bilby
from scipy.interpolate import CubicSpline
from scipy.special import erf
import shutil

class TimeSeries:
    def __init__(self,times,data):
        self.times=times
        self.t0=times[0]
        self.deltaT=times[1]-times[0]
        self.Fs=1/self.deltaT
        self.data=data
        
    def window(self):
        return TimeSeries(self.times,np.hanning(len(self.data)) * self.data)
        
    def zero_pad(self,zpf=2):
        NZeros = (zpf-1) * len(self.data)
        new_data=np.append(self.data,np.zeros(NZeros))
        additional_times= np.arange(self.times[-1],
                             self.times[-1]+NZeros*self.deltaT,
                             self.deltaT)
        new_times=np.append(self.times,additional_times)
        return TimeSeries(new_times,new_data)
            
    def window_and_fft(self):
        # window
        ts_w = self.window()
    
        # zero pad
        ts_wz = ts_w.zero_pad(zpf=1)
    
        # fft
        data_tilde = np.fft.rfft(ts_wz.data) * self.deltaT
        data_tilde=data_tilde[1:]
        
        # construct the frequency array
        deltaF=1/(len(ts_wz.data) * ts_wz.deltaT)
        fmin=deltaF
        fmax=1/(2*ts_wz.deltaT)        
        epsilon=deltaF/100
        freqs=np.arange(fmin,fmax+epsilon,deltaF)
        print('num freqs (coarse graining) =%u'%len(freqs))
        
        return FrequencySeries(freqs,data_tilde)
    
class FrequencySeries:
    def __init__(self,freqs,data):
        self.deltaF=freqs[1]-freqs[0]
        self.freqs=freqs
        self.data=data
        
        
    
    def __mul__(self,x):
        if type(x) is float or type(x) is int:
            return FrequencySeries(self.freqs,self.data*x)
        return FrequencySeries(self.freqs,self.data*x.data)
    
    def __truediv__(self,x):
        if type(x) is float or type(x) is int:
            return FrequencySeries(self.freqs,self.data/x)
        return FrequencySeries(self.freqs,self.data/x.data)
        
    def coarse_grain(self,newDeltaF):
        fmin=self.freqs[0]
        fmax=self.freqs[-1]
        epsilon=self.deltaF/100.
        new_freqs=np.arange(fmin,fmax,newDeltaF)
        new_data_real=np.zeros(len(new_freqs))
        new_data_imag=np.zeros(len(new_freqs))
        NBinsToCombine = int(newDeltaF/self.deltaF)
                
        # hack to make coarse graining work
        # add zeros to edges of original frequency series
        zero_padded_freq_min = self.freqs[0] - newDeltaF/2
        zero_padded_freq_max = self.freqs[-1] + newDeltaF/2
        zero_padded_freqs = np.arange(zero_padded_freq_min,
                                     zero_padded_freq_max+epsilon,
                                     self.deltaF)

        zero_padded_data = np.insert(self.data,0,np.zeros(int(NBinsToCombine / 2)))
        zero_padded_data = np.append(zero_padded_data,np.zeros(int(NBinsToCombine / 2)))

        zero_padded_frequency_series = FrequencySeries(zero_padded_freqs,
                                                      zero_padded_data)
      
        for ii in range(len(new_freqs)):          
            istart = ii * NBinsToCombine 
            iend = istart + NBinsToCombine
            
            new_data_real[ii] = np.mean(np.real(
                zero_padded_frequency_series.data[istart:iend]))
            new_data_imag[ii] = np.mean(np.imag(
                zero_padded_frequency_series.data[istart:iend]))

        return FrequencySeries(new_freqs,new_data_real+1j*new_data_imag)
    
def slice_time_series(timeseries,istart,iend):
    return TimeSeries(timeseries.times[istart:iend],
                      timeseries.data[istart:iend])
    

def welch_psd(data,window='hann',nperseg=None,fs=None):
    '''
    Inputs
    * data (assumed to have a length that is a power of 2)
    * window (take to be hann)
    * nperseg = Length of each segment in the PSD (TAvg/deltaT=TAvg*Fs)
    * fs = sampling rate in Hz    
        
    Output
    * f = frequency array [from 0 to f_nyquist by df = 1/TAvg]
    * psd = the psd
    '''
    
    # useful quantities
    NSamples=len(data)
    NAvgs = 2 * int(NSamples / nperseg) - 1 # 50% overlapping
    deltaT = 1/fs
    TAvg = nperseg * deltaT
    stride=int(nperseg/2) # 50% overlapping
    fNyquist = fs/2
    
    # frequency array
    fmin=0
    deltaF=1/TAvg
    fmax = fNyquist
    epsilon=deltaF/100
    freqs = np.arange(fmin,fmax+epsilon,deltaF)    
    
    # psd estimate
    psd=np.zeros(len(freqs))
    
    for nn in range(NAvgs): ## nn goes from 0 to NAvgs-1
        # slice data 
        istart = stride * nn
        iend   = istart + nperseg
        subdata = data[istart:iend]
        
        # window
        subdata = np.hanning(nperseg) * subdata
        
        # fft
        subdataTilde = np.fft.rfft(subdata) * deltaT
        
        # psd = |fft|^2 * 2 / T
        psd = psd + 1/NAvgs * (np.abs(subdataTilde)**2 * 2 / TAvg)
        
    windowFactor =  1/nperseg *  ( np.sum(np.hanning(nperseg)**2) )
    psd=psd/windowFactor
    
    freqs=freqs[:-1]
    psd=psd[:-1]
    
    return freqs,psd    

def window_factors(N):
    '''
    calculate window factors for a hann window
    '''
    w=np.hanning(N)
    w1w2bar=np.mean(w**2)
    w1w2squaredbar=np.mean(w**4)
    
    w1=w[int(N/2):N]
    w2=w[0:int(N/2)]
    w1w2squaredovlbar=1/(N/2.) * np.sum(w1**2*w2**2)
    
    w1w2ovlbar=1/(N/2.) * np.sum(w1*w2)

    return w1w2bar,w1w2squaredbar,w1w2ovlbar,w1w2squaredovlbar

def calc_Y_sigma_from_Yf_varf(Y_f,var_f,freqs=None,alpha=0,fref=1):
    if freqs is not None:
        weights = (freqs/fref)**alpha
    else:
        weights=np.ones(Y_f.shape)
    
    var = 1 / np.sum(var_f**(-1) * weights**2)
    
    #Y = np.sum(Y_f * var_f**(-1)) / np.sum( var_f**(-1) )
    Y = np.sum(Y_f * weights * (var/var_f) ) 
    
    sigma=np.sqrt(var)
    
    return Y,sigma

def calc_rho1(N):
    w1w2bar,_,w1w2ovlbar,_=window_factors(100000)
    rho1 = (0.5 * w1w2ovlbar / w1w2bar)**2
    return rho1
def calc_bias(segmentDuration,deltaF,deltaT):
    N=int(segmentDuration/deltaT)
    rho1=calc_rho1(N)
    Nsegs=(2 * segmentDuration * deltaF - 1)
    wfactor = (1+2*rho1)**(-1)
    Neff = 2*wfactor*(2*segmentDuration*deltaF-1)
    bias=Neff/(Neff-1)
    return bias

def cleanup_dir(outdir):
    # cleanup
    try:
        shutil.rmtree(outdir)
    except OSError as e:
        pass # directory doesn't exist
