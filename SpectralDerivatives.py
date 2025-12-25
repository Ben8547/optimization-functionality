'''
Author: Ben Campbell

Purpose: To implement spectral differenttation methods
'''

from numpy import ndarray, complex128, pi, multiply, exp, array, linspace, dot
from numpy import fft

import matplotlib.pyplot as plt

def Fourier_Derivative(func, x0: float, h:float ,n: int = 100000) -> float:
    '''
    Docstring for Fourier_Derivative
    Should run with an even number of sample points. Accuracy seems to be off otherwise.
    
    :param func: Function to differentiate
    :param x0: point at which to take the derivative; should modify function to take an array
    :type x0: ndarray
    :param h: separation of points sampled
    :type h: float
    :param n: number of points to sample - should be large
    :type n: int
    :return: the derivative estimate
    :rtype: float
    '''
    pointsToInterpolate = linspace(x0 - h*((n-1)/2), x0 + h * ((n-1)/2), n)
    L = pointsToInterpolate[-1] - pointsToInterpolate[0] # width of the interpolation interval
    f_vals = func(pointsToInterpolate)
    k = 2*pi * fft.fftfreq(n, d=h)
    Deriv_vals = fft.ifft(1j * k * fft.fft(f_vals)).real
    '''if __name__ == "__main__": # debugging
        plt.plot(pointsToInterpolate,Deriv_vals)
        plt.show()
        plt.plot(pointsToInterpolate,f_vals,ls="--")
        plt.show()'''
    # Now we find the interpolation points closest to x0 
    if n % 2 != 0: # n is odd means x0 is in the interpolation points
        return Deriv_vals[(n-1)//2]
    else: 
        return 0.5 * (Deriv_vals[n//2 - 1] + Deriv_vals[n//2])
    

from scipy.special import chebyt # import the chebychev polynomials

def ChebyshevDifferentiation(func):
    '''
    Docstring for ChebyshevDifferentiation
    
    
    :param func: The objective function to be differentiated
    '''
    return


# test

if __name__ == '__main__':
    f = lambda x: exp(x) # test function
    print(Fourier_Derivative(f,1,1e-10,2**14 + 3))
    print(Fourier_Derivative(f,1,1e-10,2**14 + 1))
    print(Fourier_Derivative(f,1,1e-10,2**14 + 2))
    # seems to give good results when n is even and horrible results when n is odd when f(x) = e**x (for both x=0 and x=1)
    # seems to give the best results when n is 2**14 + 1 when f(x) = e**(-x**2)
    # since most pathologies of the function are at the endpoints even when f is not periodic, this method seems alright. Not a fast as a finite difference though.
