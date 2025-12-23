'''
Author: Ben Campbell

Purpose: Functions related to polynomial interpolation
'''

from numpy import prod, ndarray, dot, zeros, float64, ones_like, asarray, empty, any, atleast_1d, where, sum, delete

def interpolatingPolynomial( x:ndarray, y:ndarray ):
    # x is an array of abscissal data points to interpolate
    # y is an array of ordinate data to interpolate
    n = len(x) # number of data points
    def interp_poly(t): # in general t is an array of inputs
        m = len(t) # number of points to evaluate at
        terms_rows = [  prod( [ ( ( t - x[i] ) / ( x[k] - x[i] ) ) if i != k else ones_like(t,dtype=float64) for i in range(n) ], axis=0 ) for k in range(n) ]
        # each array in term_rows represents the interpolation polynomial at all input values of t.
        # there are n elements of the list, each m long
        terms = zeros((n, m),dtype=float64) # each column of this row contains the value of the interpolation polynomial components for a single t
        for i in range(n):
            terms[i,:] = terms_rows[i] # setting the values of the
        return dot(y,terms) # multiplies each term by the y value and then sums; does this column wise. We get a vector back, each element corresponds to a difffernt t
    return interp_poly # returns a function

import numpy as np
from numpy import ndarray, float64, ones_like, zeros, prod

def interpolatingPolynomial_derivative(x: ndarray, y: ndarray):
    x = np.asarray(x, dtype=float64)
    y = np.asarray(y, dtype=float64)
    n = len(x)

    def dp(t):
        t = np.atleast_1d(t).astype(float64)
        m = len(t)
        out = np.zeros(m, dtype=float64)

        for k in range(n): # the for loops are not ideal but this got complicated so it was easier to visualize this way
            denom = np.prod(x[k] - np.delete(x, k))
            for j in range(n):
                if j == k:
                    continue
                term = np.ones(m, dtype=float64)
                for i in range(n):
                    if i == k or i == j:
                        continue
                    term *= (t - x[i]) / (x[k] - x[i])
                out += y[k] * term / (x[k] - x[j])

        return out

    return dp


    return interp_poly_deriv



def NDinterpolatingPolynomial(x:ndarray): # finish later
    # an interpolating polynomial for data in N dimensions
    # now x[i,:] contains the corrdinate of the data in the ith dimension
    # for example, in 3D, if the point (1,3,6) were in the set then x[0,0] = 1, x[1,0] = 3, x[2,0] = 6.
    return