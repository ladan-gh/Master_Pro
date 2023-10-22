#Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import pandas as pd
import scipy as sp
from scipy.io.wavfile import read

#-------------------------------------------------------------------------------------
(Frequency, array) = read('C:/Users/Ladan_Gh/Desktop/Voice.wav') # Reading the sound file.

#-------------------------------------------
plt.plot(array)
plt.title('Original Signal Spectrum')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#-------------------------------------------
a,b = signal.cheby2(10, 10, 15, 'hp', fs=1000)
filteredSignal_01 = signal.lfilter(a,b, array)
plt.plot(filteredSignal_01) # plotting the signal
plt.title('Highpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#-------------------------------------------
c,d = signal.cheby2(10, 1, 15, 'lp', fs=1000)
filteredSignal_02 = signal.lfilter(c,d, array)
plt.plot(filteredSignal_02) # plotting the signal
plt.title('Lowpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#---------------------------------------------
e,f = signal.cheby2(40, 1.025, [0.15, 0.95], btype='bandpass') # ButterWorth Bnad-filter
newFilteredSignal = signal.lfilter(e,f,array) # Applying the filter to the signal
plt.plot(newFilteredSignal) # plotting the signal.
plt.title('Bandpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()