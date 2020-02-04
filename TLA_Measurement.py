# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:39:24 2020

@author: Robi
"""
import numpy as np
from scipy import signal
import scipy.interpolate as interp
import scipy.fftpack

def PstLM_calculation(Waveform,Samplerate):
    if Samplerate < 2000:
        print('Sampling frequency should be >= 2000 Hz')
    fs = Samplerate
    u_0 = Waveform / np.mean(Waveform)

    HIGHPASS_ORDER  = 1
    HIGHPASS_CUTOFF = 0.05
    LOWPASS_CUTOFF  = 35     
    LOWPASS_ORDER   = 6
    u_0_ac = u_0 - np.mean(u_0)

    b_hp, a_hp = signal.butter(HIGHPASS_ORDER, HIGHPASS_CUTOFF/(fs/2), 'high') 
    u_hp = signal.lfilter(b_hp, a_hp, u_0_ac)

    smooth_limit = min(round(fs / 10), len(u_hp))
    u_hp[0:smooth_limit] = u_hp[0:smooth_limit] * np.linspace(0, 1, smooth_limit)
    b_bw, a_bw = signal.butter(LOWPASS_ORDER, LOWPASS_CUTOFF/(fs/2), 'low') 
    u_bw = signal.lfilter(b_bw, a_bw, u_hp)

    B1 = 0.041661
    B2 = 44.758
    B3 = 2715.6
    B4 = 29839
    B5 = 0
    A1 = 1
    A2 = 196.32
    A3 = 11781
    A4 = 534820
    A5 = 3505380
    num1 = B1, B2, B3, B4, B5
    den1 = A1, A2, A3, A4, A5

    b_w, a_w = signal.bilinear(num1, den1, fs)
    u_w = signal.lfilter(b_w, a_w, u_bw)

    SCALING_FACTOR   =  1.101603155420234e+06
    u_q = u_w * u_w

    LOWPASS_2_ORDER  = 1
    LOWPASS_2_CUTOFF = 1 / (2 * np.pi * 300e-3)
    b_lp, a_lp = signal.butter(LOWPASS_2_ORDER, LOWPASS_2_CUTOFF/(fs/2), 'low')

    s = SCALING_FACTOR * signal.lfilter(b_lp, a_lp, u_q)
    #P_inst = s
 
    tau_transient = 20
    n_transient = int(tau_transient * fs) - 1
    #P_inst_max = max(P_inst[n_transient:])
    
    s=s[n_transient:] 
    
    NUMOF_CLASSES = 10000   
    cpf_cum_probability, cpf_magnitude = np.histogram(s, NUMOF_CLASSES)
    cpf_cum_probability = 100 * (1 - np.cumsum(cpf_cum_probability) / sum(cpf_cum_probability))
    
    arr1_interp = interp.interp1d(np.arange(cpf_magnitude.size),cpf_magnitude)
    cpf_magnitude = arr1_interp(np.linspace(0,cpf_magnitude.size-1,cpf_cum_probability.size))

    p_50s = np.mean([get_percentile(cpf_cum_probability, cpf_magnitude, 30), get_percentile(cpf_cum_probability, cpf_magnitude, 50), get_percentile(cpf_cum_probability, cpf_magnitude, 80)])
    p_10s = np.mean([get_percentile(cpf_cum_probability, cpf_magnitude, 6), get_percentile(cpf_cum_probability, cpf_magnitude, 8), get_percentile(cpf_cum_probability, cpf_magnitude, 10),
                     get_percentile(cpf_cum_probability, cpf_magnitude, 13), get_percentile(cpf_cum_probability, cpf_magnitude, 17)])
    p_3s = np.mean([get_percentile(cpf_cum_probability, cpf_magnitude, 2.2), get_percentile(cpf_cum_probability, cpf_magnitude, 3), get_percentile(cpf_cum_probability, cpf_magnitude, 4)])
    p_1s = np.mean([get_percentile(cpf_cum_probability, cpf_magnitude, 0.7), get_percentile(cpf_cum_probability, cpf_magnitude, 1), get_percentile(cpf_cum_probability, cpf_magnitude, 1.5)])
    p_0_1 = get_percentile(cpf_cum_probability, cpf_magnitude, 0.1)

    PstLM = np.sqrt(0.0314 * p_0_1 + 0.0525 * p_1s + 0.0657 * p_3s + 0.28 * p_10s + 0.08 * p_50s)
    return PstLM

def get_percentile(cpf_cum_probability, cpf_magnitude, limit):
    val = min(abs(cpf_cum_probability - limit))
    idx = np.where(abs(cpf_cum_probability - limit) == val)
    res = cpf_magnitude[idx[0]] 
    return res[0]

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def getSensitivityStrobo(frequency):
    a = 0.00518
    b = 306.6
    getSensitivityStrobo = (1 / (1+np.exp(-a*(frequency-b))))+np.exp(-frequency/10)*20
    return getSensitivityStrobo

def stroboVisibilityMeasure(waveForm, samplerate):
    Ls = len(waveForm)
    Fs = samplerate
    Df = samplerate / Ls
    duration = 1 / samplerate * Ls
    Fmax=Ls*Df/2
    FmaxSVM = 2000
    if (Fs < FmaxSVM * 2):
        print('ERROR: Sampling frequency should be >= ', 2*FmaxSVM)
    if (any(x < 0 for x in waveForm)):
        print('ERROR: Input light waveform contains negative values! Values must be >= 0!')
    if duration < 1:
        print('ERROR:Duration of the light waveform is recommended to be at least 1 sec!')
    NFFT = nextpow2(Ls)*2
    attHann = sum(scipy.signal.windows.hann(Ls))/Ls
    window = scipy.signal.windows.hann(Ls)
    Y = np.divide(np.multiply(scipy.fftpack.fft(waveForm*window,NFFT),(1/attHann)),Ls)
    mag0= np.abs(Y[0])
    mag = np.abs(Y[:NFFT//2]) / mag0
    f = Fmax * np.linspace(0, 1, NFFT//2)
    FmaxSVMindex = round((NFFT//2)*(FmaxSVM/Fmax))
    ff = f[1:(1+FmaxSVMindex)]
    b = np.multiply(mag[1:(1+FmaxSVMindex)],2)
    peaks, _ = scipy.signal.find_peaks(b)
    peaksvalue = b[peaks]
    lenpeaks = len(peaks)
    tempff = [None] * lenpeaks
    for i in range(0,lenpeaks,1):
        tempff[i]=(ff[i]*(peaks[i]+1))/(i+1)
    peaksvalue = np.array([peaksvalue])
    tempff = np.array([tempff])
    c_pks=np.divide(peaksvalue,getSensitivityStrobo(tempff))
    SVM = pow(np.sum(np.power(c_pks,3.7)),1/3.7)
    return SVM

def modulation_depth(Waveform):
    ymax=max(Waveform)
    ymin=min(Waveform)
    mod_depth = (ymax - ymin) / (ymax + ymin) * 100
    return mod_depth

def flicker_index(Waveform):
    average = np.average(Waveform)
    full = np.trapz(Waveform)
    a1=0
    for i in range(0,len(Waveform),1):
        if Waveform[i] > average:
            a1 = a1 + (Waveform[i] - average)
    a2 = full - a1
    fi = a1 / (a1 + a2)
    return fi