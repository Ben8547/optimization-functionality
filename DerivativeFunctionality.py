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

def FnD_IP(func,x,h,n:int) -> float: # finite difference approximation of the derivaitve by fitting to degree n polynomial
    '''
    in theory this works, in practice it does not. this was my first naive approach to a degree n approximation.
    The interpolating polynomials oscilate between the points in general so the derivatives pick up unrealistic values.
    Interestingly this oscilatory effect is exagerated with higher degree approximations so the accuracy actually decreases as n gets larger.
    '''
    # n is the number of sample points. n-1 is the the degree of polynomial
    pointsToInterpolate = linspace(x - h*((n-1)/2), x + h * ((n-1)/2), n) # generates n evenly space points between x-(n-1)h/2 and x+(n-1)h/2

    # will return 0 unless n > 1.
    poly_d = interpolatingPolynomial_derivative(pointsToInterpolate,func(pointsToInterpolate)) # a degree n-1 polynomial that passes through all n points
    # now we find the derivative of the polynomial at the specified point
    return poly_d(x)

from numpy import dot, array, ones
from NonlinearSolvers import SecantSolve

def FnD(func,x,h,n:int) -> float:
    '''
    since the previous function finds the entire interpolating polynomial, there are a lot of computations that compount the rounding error.
    We might be able to do better by just finding the coeficients in th finite difference formula.
    '''
    # n is the number of sample points. n-1 is the the degree of polynomial
    # the above are picked as to cancel out terms in the taylor series
    # We have some polynomial given by a_0 + a_1 (x-x_0) + ... + a_n (x-x_0)^n
    # Then the derivative at x_0 is simply a_1, so we need only solve for a_1

    pointsToInterpolate = linspace(x - h*((n-1)/2), x + h * ((n-1)/2), n) # generates n evenly space points between x-(n-1)h/2 and x+(n-1)h/2

    A = ones(n-1) # initialize the vector of coeficients [a_0,a_1,...,a_n]

    def poly(A,t): return dot(array( [ (t-x)**i for i in range(n-1) ] ),A)

    objective_func = lambda A: poly(A, pointsToInterpolate) - func(pointsToInterpolate)

    # now we use Newton's method on the multidimensional function of A poly to find what A must be to solve the system
    # poly(A,t) = f(t) <==> poly(A,t) - f(t) = 0
    SecantSolve(objective_func,)

    return

def F6D (func,x,h): # degree 5 approximation - error of order 6 - needs 6 points
    return (-3*func(x-5*h/2)/640 + func(x-3*h/2)*25/384 -75*func(x-h/2)/64 + 75*func(x+h/2)/64 - func(x+3*h/2)*25/384 + 3*func(x+5*h/2)/640 ) / h

from numpy import ndarray


# ----------------------
# Partial Derivatives
# ----------------------

from numpy import copy

def partial_FCD(func,n:int,x:ndarray,h)-> ndarray:
    # n is the coordinate in which we take the partial derivative
    '''
    Assume f is a function R^n -> R^k
    Assume we want to compute l derivatives at once
    In: l by n matrix, each row of the matrix is a vector in the domain of f.
    Out: l by k matrix, each row of the matrix is a vector in the codomain of the partial of f.
    '''
    if len(x.shape) == 1: # convert vector to matrix
        x_ = x.reshape((1,len(x))) # 1 by n matrix
    else: x_ = copy(x)

    left = copy(x_) # this is a matrix (lxn)
    right = copy(x_)
    left[:,n] = x_[:,n]+h/2 # take the n^th entry of each row and replace
    right[:,n] = x_[:,n]-h/2
    return (func(left) - func(right)) / h # this is an l by k matrix


 