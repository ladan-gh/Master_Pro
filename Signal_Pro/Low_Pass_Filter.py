# Link: https://github.com/joaossmacedo/Low_Pass_Filter/blob/master/Low_Pass_Filter.ipynb

#---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import math
import contextlib
import numpy as np
import wave
import struct

#-----------------------------------------------------------------------------------------
fname = "C:/Users/Ladan_Gh/Desktop/Recording.wav"
outname = 'filtered.wav'

#----------------------------Function to interprete .wav------------------------------------
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.frombuffer(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels

#--------------------Running mean------------------------------------------
def running_mean(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0))
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

#--------------------Filter .wav--------------------------------------------
cutOffFrequency = 10.0

with contextlib.closing(wave.open(fname, 'rb')) as spf:
    # sampling frequency
    sampleRate = spf.getframerate()
    # width in bytes
    ampWidth = spf.getsampwidth()
    # mono(1) or stereo(2)
    nChannels = spf.getnchannels()
    # number of audio frames
    nFrames = spf.getnframes()

    # Extract raw audio from multi-channel .wav file
    # returns at most nFrames*nChannels as bytes object
    signal = spf.readframes(nFrames * nChannels)
    spf.close()
    channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

    # get window size
    freqRatio = (cutOffFrequency / sampleRate)
    N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)

    # Use moving average (only on first channel)
    filtered = running_mean(channels[0], N).astype(channels.dtype)

    # Writes the result
    wav_file = wave.open(outname, "w")
    # getcomptype and getcompname returns the compression type
    wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
    wav_file.writeframes(filtered.tobytes('C'))
    wav_file.close()

#--------------Show initial wave----------------------
spf = wave.open(fname,'r')

signal = spf.readframes(-1)
signal = np.frombuffer(signal, 'int16')

fs = spf.getframerate()
Time=np.linspace(0, len(signal)/fs, num=len(signal))

plt.figure(1)
plt.title('Initial Wave')
plt.plot(Time, signal)
plt.show()

#--------------Show filtered wave----------------------
spf = wave.open(outname,'r')

signal = spf.readframes(-1)
signal = np.frombuffer(signal, 'int16')

fs = spf.getframerate()
Time=np.linspace(0, len(signal)/fs, num=len(signal))

plt.figure(1)
plt.title('Filtered Wave')
plt.plot(Time, signal)
plt.show()