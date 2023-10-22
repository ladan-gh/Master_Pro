import numpy as np
import scipy.integrate as spi

#t is the independent variable
P = 3. #period value
BT=-6. #initian value of t (begin)
ET=6. #final value of t (end))
FS=1000 #number of discrete values of t between BT and ET

#the periodic real-valued function f(t) with period equal to P to simulate an acquired dataset
f = lambda t: ((t % P) - (P / 2.)) ** 3
t_range = np.linspace(BT, ET, FS) #all discrete values of t in the interval from BT and ET
y_true = f(t_range)

#function that computes the real fourier couples of coefficients (a0, 0), (a1, b1)...(aN, bN)
def compute_real_fourier_coeffs_from_discrete_set(y_dataset, N):
    result = []
    T = len(y_dataset)
    t = np.arange(T)
    for n in range(N+1):
        an = (2./T) * (y_dataset * np.cos(2 * np.pi * n * t / T)).sum()
        bn = (2./T) * (y_dataset * np.sin(2 * np.pi * n * t / T)).sum()
        result.append((an, bn))
    return np.array(result)

#function that computes the real form Fourier series using an and bn coefficients
def fit_func_by_fourier_series_with_real_coeffs(t, AB):
    result = 0.
    A = AB[:,0]
    B = AB[:,1]
    for n in range(0, len(AB)):
        if n > 0:
            result +=  A[n] * np.cos(2. * np.pi * n * t / P) + B[n] * np.sin(2. * np.pi * n * t / P)
        else:
            result +=  A[0]/2.
    return result

FDS=20. #number of discrete values of the dataset
t_period = np.arange(0, P, 1/FDS)
y_dataset = f(t_period) #generation of discrete dataset

maxN=8
#print in the range from BT to ET, the f(t)
for N in range(1, maxN + 1):
    AB = compute_real_fourier_coeffs_from_discrete_set(y_dataset, N)
    #AB contains the list of couples of (an, bn) coefficients for n in 1..N interval.
    print(AB)

    y_approx = fit_func_by_fourier_series_with_real_coeffs(t_range, AB)
    #y_approx contains the discrete values of approximation obtained by the Fourier series

#---------------------------------------------------
# # Fourier Series Coefficients
# # The following function returns the fourier coefficients,'a0/2', 'An' & 'Bn'
# #
# # User needs to provide the following arguments:
# #
# # l=periodicity of the function f which is to be approximated by Fourier Series
# # n=no. of Fourier Coefficients you want to calculate
# # f=function which is to be approximated by Fourier Series
# #
# # *Some necessary guidelines for defining f:
# # *The program integrates the function f from -l to l so make sure you define the function f correctly in the interval -l to l.
# #
# # for more information on Fourier Series visit: https://en.wikipedia.org/wiki/Fourier_series
# #
# # Written by: Manas Sharma(manassharma07@live.com)
# # For more useful toolboxes and tutorials on Python visit: https://www.bragitoff.com/category/compu-geek/python/
#
# def fourier(li, lf, n, f):
#     l = (lf - li) / 2
#     # Constant term
#     a0 = 1 / l * integrate.quad(lambda x: f(x), li, lf)[0]
#     # Cosine coefficents
#     A = np.zeros((n))
#     # Sine coefficents
#     B = np.zeros((n))
#
#     for i in range(1, n + 1):
#         A[i - 1] = 1 / l * integrate.quad(lambda x: f(x) * np.cos(i * np.pi * x / l), li, lf)[0]
#         B[i - 1] = 1 / l * integrate.quad(lambda x: f(x) * np.sin(i * np.pi * x / l), li, lf)[0]
#
#     return [a0 / 2.0, A, B]
#
#
# # Author: Manas Sharma
# # Website: www.bragitoff.com
# # Email: manassharma07@live.com
# # License: MIT
#
# import numpy as np
# import scipy.integrate as integrate
#
#
# # Non-periodic sawtooth function defined for a range [-l,l]
# def sawtooth(x):
#     return x
#
#
# # Non-periodic square wave function defined for a range [-l,l]
# def square(x):
#     if x > 0:
#         return np.pi
#     else:
#         return -np.pi
#
#
# # Non-periodic triangle wave function defined for a range [-l,l]
# def triangle(x):
#     if x > 0:
#         return x
#     else:
#         return -x
#
#
# # Non-periodic cycloid wave function defined for a range [-l,l]
# def cycloid(x):
#     return np.sqrt(np.pi ** 2 - x ** 2)
#
#
# # Fourier Series Coefficients
# # The following function returns the fourier coefficients,'a0/2', 'An' & 'Bn'
# #
# # User needs to provide the following arguments:
# #
# # l=periodicity of the function f which is to be approximated by Fourier Series
# # n=no. of Fourier Coefficients you want to calculate
# # f=function which is to be approximated by Fourier Series
# #
# # *Some necessary guidelines for defining f:
# # *The program integrates the function f from -l to l so make sure you define the function f correctly in the interval -l to l.
# #
# # for more information on Fourier Series visit: https://en.wikipedia.org/wiki/Fourier_series
# #
# # Written by: Manas Sharma(manassharma07@live.com)
# # For more useful tutorials on Python visit: https://www.bragitoff.com/category/compu-geek/python/
# def fourier(li, lf, n, f):
#     l = (lf - li) / 2
#     # Constant term
#     a0 = 1 / l * integrate.quad(lambda x: f(x), li, lf)[0]
#     # Cosine coefficents
#     A = np.zeros((n))
#     # Sine coefficents
#     B = np.zeros((n))
#
#     for i in range(1, n + 1):
#         A[i - 1] = 1 / l * integrate.quad(lambda x: f(x) * np.cos(i * np.pi * x / l), li, lf)[0]
#         B[i - 1] = 1 / l * integrate.quad(lambda x: f(x) * np.sin(i * np.pi * x / l), li, lf)[0]
#
#     return [a0 / 2.0, A, B]
#
#
# if __name__ == "__main__":
#     # Limits for the functions
#     li = -np.pi
#     lf = np.pi
#
#     # Number of harmonic terms
#     n = 3
#
#     # Fourier coeffficients for various functions
#     coeffs = fourier(li, lf, n, sawtooth)
#     print('Fourier coefficients for the Sawtooth wave\n')
#     print('a0 =' + str(coeffs[0]))
#     print('an =' + str(coeffs[1]))
#     print('bn =' + str(coeffs[2]))
#     print('-----------------------\n\n')
#
#     coeffs = fourier(li, lf, n, square)
#     print('Fourier coefficients for the Square wave\n')
#     print('a0 =' + str(coeffs[0]))
#     print('an =' + str(coeffs[1]))
#     print('bn =' + str(coeffs[2]))
#     print('-----------------------\n\n')
#
#     coeffs = fourier(li, lf, n, triangle)
#     print('Fourier coefficients for the Triangular wave\n')
#     print('a0 =' + str(coeffs[0]))
#     print('an =' + str(coeffs[1]))
#     print('bn =' + str(coeffs[2]))
#     print('-----------------------\n\n')
#
#     coeffs = fourier(li, lf, n, cycloid)
#     print('Fourier coefficients for the Cycloid wave\n')
#     print('a0 =' + str(coeffs[0]))
#     print('an =' + str(coeffs[1]))
#     print('bn =' + str(coeffs[2]))
#     print('-----------------------\n\n')
#
# #https://www.bragitoff.com/2021/05/fourier-series-coefficients-and-visualization-python-program/

#----------------------------------------