#Link: https://www.sololearn.com/compiler-playground/cJinz7E6LHeZ/
import numpy as np
import mpmath as mp
import scipy
import scipy.stats as sp
import matplotlib.pyplot as plt
import subprocess
import cmath as cm

#-------------------------------------------------
def dtft(x1, N):
    x = x1[0]
    j = cm.sqrt(-1)
    n = x1[1]

    w = np.linspace(-np.pi, np.pi, N)
    for i in range(0, N):
        w_tmp = w[i]
        X_tmp = 0
        for k in range(0, len(x)):

            X_tmp += (x[k] * np.exp(-n[k] * w_tmp * j))

        print(X_tmp) #Print DTFT result



#------------------------------------------------------
# Example
x = [1 / 2, 1 / 2]
n = [0, 1]
x1 = [x, n]
dtft(x1, 100)
