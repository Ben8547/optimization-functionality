"""
Author: Benjamin Campbell

Purpose: Implement a library of derivative functions for use in other applications.
I do not return error estimates since they require knowledge of higher order derivatives. This can be estimated through other means
when the function is applied.
"""

def FFD(func,x,h) -> float:
    return (func(x+h) - func(x))/h

def FBD(func,x,h) -> float: return (func(x) - func(x-h))/h

def FCD(func,x,h) -> float: return ( func(x+h) - func(x-h) ) / (2.*h)

from numpy import linspace
from InterpolatingPolynomials import interpolatingPolynomial_derivative

def FnD(func,x,h,n:int) -> float: # finite difference approximation of the derivaitve by fitting to degree n polynomial
    '''
    in theory this works, in practice it does not. this was my first naive approach to a degree n approximation.
    The interpolating polynomials oscilate between the points in general so the derivatives pick up unrealistic values.
    Interesting this oscilatory effect is exagerated with higher degree approximations so the accuracy actually decreases as n gets larger.
    '''
    # n is the number of sample points. n-1 is the the degree of polynomial
    pointsToInterpolate = linspace(x - h*((n-1)//2), x + h * ((n-1)//2), n) # generates n evenly space points between x-h and x+h

    # will return 0 unless n > 1.
    poly_d = interpolatingPolynomial_derivative(pointsToInterpolate,func(pointsToInterpolate)) # a degree n-1 polynomial that passes through all n points
    # now we find the derivative of the polynomial at the specified point
    return poly_d(x)


 