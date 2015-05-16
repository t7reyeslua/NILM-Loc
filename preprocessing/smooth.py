# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:39:09 2015

@author: t7

smooth function taken from http://wiki.scipy.org/Cookbook/SignalSmooth
"""

import numpy
from pandas import Series

def smooth(serie,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    x = serie.values
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    #return y
    y = y[(window_len/2-1):-(window_len/2)]
    ss = Series(y, index = serie.index)
    return ss


def when_on(serie, on_power_threshold):   
    res = []
    for value in serie.values:
        res.append(value>on_power_threshold)
    ss = Series(res, index=serie.index)
    return ss

        
def calculate_events(serie):
    """
    Get only the timestamp of when each ON/OFF event for each appliance 
    was triggered.
    """
    
    # Detect the moment at which there was a transition for each appliance
    # For this, we shift the whole vector and compare when two consecutive values are different.

    current_state = serie
    previous_state = current_state.shift()
    trigger_events = current_state[current_state != previous_state]

    return trigger_events

def fit_events(smoothed, original):
    events = {}
    for i, event in enumerate(smoothed):
        events_same_value = original[original == event]
        closest_index = numpy.argmin(numpy.abs(events_same_value.index.to_pydatetime() - smoothed.index[i]))
        closest = events_same_value.index[closest_index]
        events[closest] = event    
    ss = Series(events)
    return ss
    
def get_events_from_smoothed_signal(original, window_len=12, on_power_threshold = 100):
    smoothed = smooth(original, window_len)
    smWO = when_on(smoothed, on_power_threshold)
    orWO = when_on(original, on_power_threshold)
    
    smEV = calculate_events(smWO)
    orEV = calculate_events(orWO)
    
    smFE = fit_events(smEV,orEV)
    
    return smFE
