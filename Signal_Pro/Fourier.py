import numpy as np
import matplotlib.pyplot as plt

t = np.arange(1)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, sp.real, freq, sp.imag)
# plt.show()
print(sp)


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fftpack import fft
#
# t = np.arange(0, 10.5, step=0.5)
# time_serie = np.exp(-t)
#
# print(time_serie)
# plt.figure()
# plt.stem(t, time_serie)
# plt.grid(True)
# plt.xticks([2, 4, 6, 8, 10])
# plt.show()

#----------------------------------------