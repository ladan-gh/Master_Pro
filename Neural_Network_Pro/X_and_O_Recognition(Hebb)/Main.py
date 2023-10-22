import numpy as np
#--------------------------------
X = np.array([[1, -1, -1, -1, 1],
             [-1, 1, -1, 1, -1],
             [-1, -1, 1, -1, -1],
             [-1, 1, -1, 1, -1],
             [1, -1, -1, -1, 1]])


O = np.array([[-1, 1, 1, 1, -1],
             [1, -1, -1, -1, 1],
             [1, -1, -1, -1, 1],
             [1, -1, -1, -1, 1],
             [-1, 1, 1, 1, -1]])


w = np.zeros((5, 5))

b = 0

t1 = 1
t2 = -1

w = w + (X * t1)
b = b + t1
w = w + (O * t2)
b = b + t2

print('w_X:', w)
print('bias:', b)

#---------------Netowrk Test-----------------
X_test = np.array([[1, -1, -1, -1, 1],
                 [-1, 1, -1, 1, -1],
                 [-1, 1, -1, -1, -1], ## mid
                 [-1, 1, -1, 1, -1],
                 [1, -1, -1, -1, -1]])

O_test = np.array([[-1, 1, 1, 1, -1],
                 [1, -1, -1, -1, 1],
                 [1, -1, -1, -1, 1],
                 [1, -1, -1, -1, 1],
                 [-1, 1, 1, 1, -1]])

y_in = b + np.sum(X_test * w)

#-------------------------------
if y_in > 0:
    y_out = 1

else:
    y_out = -1

#-------------------------------
if y_out == 1:
    print('This is a X :)')
else:
    print('This is a O :/')