from scipy.io import wavfile
import scipy.signal as sig
from pylab import *
from scipy.io.wavfile import read, write
import io
from scipy import signal
import scipy.io.wavfile
import librosa

#-------------------------------------------------------------
def mfreqz(b,a=1):
    w,h = sig.freqz(b,a)
    h_dB = 20 * log10 (abs(h))
    subplot(211)
    plot(w/max(w),h_dB)
    ylim(-150, 5)
    ylabel('Magnitude (db)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Frequency response')
    subplot(212)
    h_Phase = unwrap(arctan2(imag(h),real(h)))
    plot(w/max(w),h_Phase)
    ylabel('Phase (radians)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Phase response')
    subplots_adjust(hspace=0.5)

#------------------------------------------------------------------
samplerate, data = wavfile.read('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Signal_Pro/2019-10-22-08-40_Fraunhofer-IDMT_30Kmh_63533_M_D_CL_ME_CH12.wav')

#--------------------Lowpass FIR filter------------------------------
n = 101
a = sig.firwin(n, cutoff = 0.3, window = "hamming", pass_zero=False)
mfreqz(a)
show()

#=========================================
N  = 6    # Filter order
fc = 1000/3000 # Cutoff frequency, normalized
b, a = signal.butter(N, fc)

#Apply the filter
tempf = signal.firwin(b,a, data)

#+++++++++++++++++++++++++++++++++
librosa.output.write_wav('3mod.wav',tempf,sf)