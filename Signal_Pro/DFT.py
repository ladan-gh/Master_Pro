import matplotlib.pyplot as plt
import numpy as np

t = np.arange(256)
sp = np.fft.fft(np.sin(t))

print(sp)
