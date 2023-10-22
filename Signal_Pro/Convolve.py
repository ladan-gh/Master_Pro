import numpy as np

a = [0,3,1,2,-1]
b = [3,2,1]

c = np.convolve(a,b)
print(c)