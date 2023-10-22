#Link : https://github.com/davidpraise45/Audio-Signal-Processing/blob/master/Sound-Filtering.py

import numpy as np
import scipy as sp
from scipy.io.wavfile import read
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

#-------------------------------------------------------------------------------------
(Frequency, array) = read('C:/Users/Ladan_Gh/Desktop/Voice.wav') # Reading the sound file.
# samplerate, data = wavfile.read("C:/Users/Ladan_Gh/Desktop/Recording.wav")

#----------------------------------------plot original signal----------------------------
plt.plot(array)
plt.title('Original Signal Spectrum')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#----------------------------------------plot highpass_filter signal----------------------------
b,a = signal.butter(5, 1000/(Frequency/2), btype='highpass') # ButterWorth filter 4350
filteredSignal = signal.lfilter(b,a,array)
plt.plot(filteredSignal) # plotting the signal.
plt.title('Highpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#----------------------------------------plot lowpass_filter signal----------------------------
c,d = signal.butter(5, 380/(Frequency/2), btype='lowpass') # ButterWorth low-filter
newFilteredSignal = signal.lfilter(c,d,filteredSignal) # Applying the filter to the signal
plt.plot(newFilteredSignal) # plotting the signal.
plt.title('Lowpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#---------------------------------------------------------------------------------------------
e,f = signal.butter(2, [0.02, 0.06], btype='bandpass', analog=False) # ButterWorth Bnad-filter
newFilteredSignal = signal.lfilter(e,f,array) # Applying the filter to the signal
plt.plot(newFilteredSignal) # plotting the signal.
plt.title('Bandpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()