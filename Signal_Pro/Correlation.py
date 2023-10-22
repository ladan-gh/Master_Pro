import librosa
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy

#---------------------------------------------------------------
data, sample_rate = librosa.load('C:/Users/Ladan_Gh/Desktop/Voice.wav')
plt.plot(data)
plt.show()

#-----------------check shape of signal and its correlate-------
r = numpy.correlate(data, data, mode='full')[len(data)-1:]
print(data.shape, r.shape)

#------------------plot Correlation-------------------------------
plt.figure(figsize=(14, 5))
plt.plot(r[:10000])
plt.title('autocorrelation Plot')
plt.xlabel('Lag (samples)')
plt.xlim(0, 10000)
plt.show()