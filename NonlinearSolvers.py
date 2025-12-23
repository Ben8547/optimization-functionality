"""
Author: Benjamin Campbell

Purpose: Implement some non-linear equation solvers (mainly Newton, Secant and Gradiant descent, I'll add relaxation or binary search later if I find a use)
"""
from numpy import ndarray, abs, max

def NewtonSolve(func,d_func,init:ndarray,error_tol:float) -> ndarray:
    error = error_tol + 1 # ensures at least one loop
    while error > error_tol:
        guess = init - d_func(init)/func(init)
        error = max(abs(init - guess))
        init = guess # update the initial vector
    return init # return the solution