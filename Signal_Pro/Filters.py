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












#-------------------------------------------------------------------
# data is a numpy ND array representing the audio data. Let's do some stuff with it
# reversed_data = a[::-1] #reversing it
#
# #then, let's save it to a BytesIO object, which is a buffer for bytes object
# bytes_wav = bytes()
# byte_io = io.BytesIO(bytes_wav)
# write(byte_io, samplerate, reversed_data)
#
# output_wav = byte_io.read() # and back to bytes, tadaaa

#Link: https://gist.github.com/hadware/8882b980907901426266cb07bfbfcd20

#--------------------Highpass FIR Filter----------------------------
# a = sig.firwin(n, cutoff = 0.3, window = "hanning", pass_zero=False)
# mfreqz(a)
# show()
#
# #-------------------Bandpass FIR filter------------------------------
# n = 1001
# a = sig.firwin(n, cutoff = [0.2, 0.5], window = 'blackmanharris', pass_zero = False)
# mfreqz(a)
# show()

#----------------------High Pass-------------------------------------
# from scipy.signal import butter, filtfilt
# import numpy as np
#
# def butter_highpass(cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return b, a
#
# def butter_highpass_filter(data, cutoff, fs, order=5):
#     b, a = butter_highpass(cutoff, fs, order=order)
#     y = filtfilt(b, a, data)
#     return y
#
# rawdata = np.loadtxt('sampleSignal.txt', skiprows=0)
# signal = rawdata
# fs = 100000.0
#
# cutoff = 100
# order = 6
# conditioned_signal = butter_highpass_filter(signal, cutoff, fs, order)
#--------------------------------------------------------------------
# import required modules
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import math
#
# # Specifications of Filter
#
# # sampling frequency
# f_sample = 3500
#
# # pass band frequency
# f_pass = 1050
#
# # stop band frequency
# f_stop = 600
#
# # pass band ripple
# fs = 0.5
#
# # pass band freq in radian
# wp = f_pass/(f_sample/2)
#
# # stop band freq in radian
# ws = f_stop/(f_sample/2)
#
# # Sampling Time
# Td = 1
#
# # pass band ripple
# g_pass = 1
#
# # stop band attenuation
# g_stop = 50
#
# # Conversion to prewrapped analog frequency
# omega_p = (2/Td)*np.tan(wp/2)
# omega_s = (2/Td)*np.tan(ws/2)
#
#
# # Design of Filter using signal.buttord function
# N, Wn = signal.buttord(omega_p, omega_s, g_pass, g_stop, analog=True)
#
#
# # Printing the values of order & cut-off frequency!
# print("Order of the Filter=", N) # N is the order
# # Wn is the cut-off freq of the filter
# print("Cut-off frequency= {:.3f} rad/s ".format(Wn))
#
#
# # Conversion in Z-domain
#
# # b is the numerator of the filter & a is the denominator
# b, a = signal.butter(N, Wn, 'high', True)
# z, p = signal.bilinear(b, a, fs)
#
# # w is the freq in z-domain & h is the magnitude in z-domain
# w, h = signal.freqz(z, p, 512)
#
#
# # Magnitude Response
# plt.semilogx(w, 20*np.log10(abs(h)))
# plt.xscale('log')
# plt.title('Butterworth filter frequency response')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Amplitude [dB]')
# plt.margins(0, 0.1)
# plt.grid(which='both', axis='both')
# plt.axvline(100, color='green')
# plt.show()
#
#
# # Impulse Response
# imp = signal.unit_impulse(40)
# c, d = signal.butter(N, 0.5)
# response = signal.lfilter(c, d, imp)
# plt.stem(np.arange(0, 40),imp,markerfmt='D',use_line_collection=True)
# plt.stem(np.arange(0,40), response,use_line_collection=True)
# plt.margins(0, 0.1)
# plt.xlabel('Time [samples]')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()
#
#
# # Phase Response
# fig, ax1 = plt.subplots()
# ax1.set_title('Digital filter frequency response')
# ax1.set_ylabel('Angle(radians)', color='g')
# ax1.set_xlabel('Frequency [Hz]')
# angles = np.unwrap(np.angle(h))
# ax1.plot(w/2*np.pi, angles, 'g')
# ax1.grid()
# ax1.axis('tight')
# plt.show()
#--------------------------------------------------------------------------------
# sampleRate, data = scipy.io.wavfile.read('C:/Users/Ladan_Gh/Desktop/2019-10-22-08-40_Fraunhofer-IDMT_30Kmh_63533_M_D_CL_ME_CH12.wav')
# times = np.arange(len(data))/sampleRate
#
# b, a = signal.butter(3, 0.05, 'lowpass')
# filteredLowPass = signal.filtfilt(b, a, data)
#
# b, a = scipy.signal.butter(3, 0.05, 'highpass')
# filteredHighPass = scipy.signal.filtfilt(b, a, data)
#
# b, a = scipy.signal.butter(3, [.01, .05], 'band')
# filteredBandPass = scipy.signal.lfilter(b, a, data)
#----------------------------------------------------------------
# from scipy import signal
# import numpy as np
# import sounddevice as sd
#
# sampling_rate = 48000 #KH
# duration_in_seconds = 2 #sec
# highpass = False
# amplitude = 0.3
#
# duration_in_samples = int(duration_in_seconds * sampling_rate)
#---------------------------------------------------------------
# from scipy import signal
# b = signal.firwin(80, 0.5, window=('kaiser', 8))
# w, h = signal.freqz(b)
# print(type(h))