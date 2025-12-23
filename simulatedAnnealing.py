'''
Author : Benjamin Campbell

Purpose: Defines a function to implement simulated annealing optomization.
'''


from numpy import exp, abs, ndarray, copy
from numpy.random import random, randint

def simulated_annealing(objective, domain_LB: ndarray, domain_UB: ndarray, tau: float, T0: float, args = [], min = True) -> float:

    '''
    :param objective: The objective ("energy" function)
    :param domain_LB: an array containing the lower bounds of the each dimension of the intervals 
    :param tau: the schedule time constant for the "tempurature". Larger is better. Each iteration of the markov chain increases time by 1 so tau should probabily be fairly large (millions is my naive guess; should also depend on the domain's dimension).
    :type tau: float
    :param T0: initial "tempurature". T0 should be much larger than a typical output of the objective function on its domain of interest 
    :type T0: float
    :param args: additional arguments of the objective function, package in a list
    :param min: Boolean - True if minimizing the objective; False if we maximize the objective
    '''
    
    if min: # if we are not minimizing then we are maximizing, so we minimize the negative of the objective function
        if args != []: # there are additional arguments specified
            objective1 = lambda x: objective(x,*args)
        else:
            objective1 =  lambda x: objective(x)
    else:
        if args != []:
            objective1 = lambda x: -objective(x,*args)
        else:
            objective1 =  lambda x: -objective(x)
    
    t = 0 # initialize the current time
    # find the initial system state
    n = len(domain_LB) # dimension of the domain
    x = random(n) #random vector in unit sphere in R^n if domain is subset of R^n
    x = (domain_UB - domain_LB) * x + domain_LB # shift and scale each component of the vector approprately to fit in the domain of the objective function
    E_x = objective1(x)
    T = copy(T0) # initialize the 'tempurature'

    while T > 1e-16: # while the 'tempurature' is not close to zero
        T = temp_schedule(t,T0,tau) # find the current 'tempurature'
        # run the markov chain
        # alter the state
        x_temp = copy(x)
        r_i = randint(0,n) # index to change
        x_temp[r_i] = domain_LB[r_i] + random() * ( domain_UB[r_i] - domain_LB[r_i] ) # change one coordinate of the domain
        #find difference in "energy"
        E_x1 = objective1(x_temp)
        E_diff = E_x1 - E_x
        #compute the acceptance probability
        P_ij = 1. if E_diff <= 0 else exp( - (E_diff) / T)
        #accept or fail the change
        if random() < P_ij:
            #accept
            x = x_temp
            E_x = E_x1
        # if reject; do nothing

        t += 1 # add one unit to the time
    return x , E_x # return the final state of the Markov chain and the value of the objective there - should be close to the global minimum of tau was large enough



def temp_schedule(t,T0,tau): return T0 * exp(-t/tau) # defines the tempurature schedule