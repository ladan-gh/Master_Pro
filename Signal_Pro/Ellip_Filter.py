#Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html

import numpy as np
import scipy as sp
from scipy.io.wavfile import read
from scipy import signal
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------
(Frequency, array) = read('C:/Users/Ladan_Gh/Desktop/Voice.wav') # Reading the sound file.

#---------------------------------------------------------------------
sos = signal.ellip(8, 1, 100, 17, 'hp', fs=1000, output='sos')
filtered_01 = signal.sosfilt(sos, array)
plt.plot(filtered_01) # plotting the signal
plt.title('Highpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#--------------------------------------------------------------------
sos = signal.ellip(8, 1, 100, 17, 'low', fs=1000, output='sos')
filtered_02 = signal.sosfilt(sos, array)
plt.plot(filtered_02) # plotting the signal
plt.title('Lowpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()

#====================================================
fs = np.array([1500, 2450])
fp = np.array([1400, 2100])
Fs = 7000
wp = fp/(Fs/2)
ws = fs/(Fs/2)

e,f = signal.ellip(2, 0.4, 50, [0.4, 0.6], btype='bandpass') # ButterWorth Bnad-filter
newFilteredSignal = signal.lfilter(e,f,array) # Applying the filter to the signal
plt.plot(newFilteredSignal) # plotting the signal.
plt.title('Bandpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()