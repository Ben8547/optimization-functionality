"""
Author: Benjamin Campbell

Purpose: Implement some non-linear equation solvers (mainly Newton, Secant and Gradiant descent, I'll add relaxation or binary search later if I find a use)
"""
from numpy import ndarray, abs, max

def NewtonSolve(func,d_func,init:ndarray,error_tol:float) -> ndarray: # this version only solves equation of 1 variablea
    error = error_tol + 1 # ensures at least one loop
    while error > error_tol:
        guess = init - d_func(init)/func(init)
        error = max(abs(init - guess))
        init = guess # update the initial vector
    return init # return the solution

def FCD(func,x,h) -> float: return ( func(x+h) - func(x-h) ) / (2.*h) # would be cylclic if we imported from the other script, I think

def SecantSolve(func,init:ndarray,error_tol:float,h=1e-10) -> ndarray: # only for equations of 1 varialbe
    error = error_tol + 1 # ensures at least one loop
    while error > error_tol:
        d_est = FCD(func,init,h)
        guess = init - d_est/func(init)
        error = max(abs(init - guess))
        init = guess # update the initial vector
    return

from numpy import zeros,copy,shape,reshape
from numpy.linalg import solve

# below here is not complete

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

def SecantSolveND(func,init:ndarray,error_tol:float,h = 1e-10) -> ndarray: # solve equations of multidimensional inputs
    '''
    We assume the system of equations is in the form f(x) = 0 where f: R^n -> R^n
    '''
    error = error_tol + 1 # ensures at least one loop
    n = len(init) # we assume func: R^n -> R^n
    while error > error_tol:
        Jacobian_est = zeros((n,n))
        for j in range(n):
            Jacobian_est[:,j] = partial_FCD(func,j,init,h) # compute the jth partial of the each equation of the system
        # this is the Jacobian at the initial point
        Delta = solve( Jacobian_est, func(init) )
        guess = init - Delta
        error = max(abs(init - guess))
        init = guess
    return init