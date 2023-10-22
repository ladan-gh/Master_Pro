from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import pandas as pd
import scipy as sp
from scipy.io.wavfile import read
# import fir1
#-------------------------------------------------------------------------------
fr, data = read('C:/Users/Ladan_Gh/Desktop/Voice.wav') # Reading the sound file.

plt.plot(data)
plt.title('Original Signal Spectrum')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#-------------------Highpass FIR Filter------------------------------------------
n = 101
a = signal.firwin(n, cutoff = 0.60, window = "hann", pass_zero=False)
b = signal.lfilter(a, 1.5, data)
plt.plot(b)
plt.title('Highpass FIR Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#-------------------Lowpass FIR filter-------------------------------------------
n = 61
a = signal.firwin(n, cutoff = 0.3, window = "hamming")
b = signal.lfilter(a, 1, data)
plt.plot(b)
plt.title('Lowpass FIR Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#-------------------Bandpass FIR filter-------------------------------------------
n = 1001
a = signal.firwin(n, cutoff = [0.2, 0.5], window = 'blackmanharris', pass_zero = False)
b = signal.lfilter(a, 1, data)
plt.plot(b)
plt.title('Bandpass FIR Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()
