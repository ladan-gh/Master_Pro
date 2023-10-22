import matplotlib.pyplot as plt
import sys
import math
import numpy as np
import wave
from scipy.io import wavfile
import scipy.signal as sig
from pylab import *
from scipy.io.wavfile import read, write
import io
from scipy import signal
import scipy.io.wavfile
import csv
import pandas as pd

fname = "C:/Users/Ladan_Gh/Desktop/Recording.wav"
samplerate, data = wavfile.read("C:/Users/Ladan_Gh/Desktop/Recording.wav")


# with open('example.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(data)
#
# f = pd.read_csv('example.csv')
# print(f)

#-------------------------------------------------------------------
# Link:https://github.com/Lukious/wav-to-csv

import sys, os, os.path
from scipy.io import wavfile
import pandas as pd

input_filename = input("Input file number:")
if input_filename[-3:] != 'wav':
    print('WARNING!! Input File format should be *.wav')
    sys.exit()

samrate, data = wavfile.read(str('C:/Users/Ladan_Gh/Desktop/' + input_filename))
print('Load is Done! \n')

wavData = pd.DataFrame(data)

if len(wavData.columns) == 2:
    print('Stereo .wav file\n')
    wavData.columns = ['R', 'L']
    stereo_R = pd.DataFrame(wavData['R'])
    stereo_L = pd.DataFrame(wavData['L'])
    print('Saving...\n')
    stereo_R.to_csv(str(input_filename[:-4] + "_Output_stereo_R.csv"), mode='w')
    stereo_L.to_csv(str(input_filename[:-4] + "_Output_stereo_L.csv"), mode='w')
    # wavData.to_csv("Output_stereo_RL.csv", mode='w')
    print('Save is done ' + str(input_filename[:-4]) + '_Output_stereo_R.csv , '
                          + str(input_filename[:-4]) + '_Output_stereo_L.csv')

elif len(wavData.columns) == 1:
    print('Mono .wav file\n')
    wavData.columns = ['M']

    wavData.to_csv(str(input_filename[:-4] + "_Output_mono.csv"), mode='w')

    print('Save is done ' + str(input_filename[:-4]) + '_Output_mono.csv')

else:
    print('Multi channel .wav file\n')
    print('number of channel : ' + len(wavData.columns) + '\n')
    wavData.to_csv(str(input_filename[:-4] + "Output_multi_channel.csv"), mode='w')

    print('Save is done ' + str(input_filename[:-4]) + 'Output_multi_channel.csv')


f = pd.read_csv('Output_multi_channel.csv')
print(f['R'])

#2019-10-22-08-40_Fraunhofer-IDMT_30Kmh_63533_M_D_CL_ME_CH12.wav