# -*- coding: utf-8 -*-
## This code is supposed to operationalize the EM algorithm published in 
### Newman 2018, Network structure from rich but noisy data, Nature Physice 14, 542-545.

import numpy as np


#fixed n
# def abr_step(n,E,Q):
#     """Run the expectation (abr) step of the EM algorithm.
#     
#     n - number of times that each edge has been measured. Numpy array.
#     E - actual observed number of edges in numpy array form.
#     Q - current estimate than edge is actually present. Numpy array.
#     """
#     # Get N (number of nodes)
#     N = E.shape[0]
#     
#     #Calculate a,b, and r
#     anum = 0
#     aden = 0
#     bnum = 0
#     bden = 0
#     r = 0
#     for j in range(1,N):
#         for i in range(j):
#             anum += E[i,j]*Q[i,j]
#             aden += Q[i,j]
#             bnum += E[i,j]*(1-Q[i,j])
#             bden += (1-Q[i,j])
#     a = anum*1./(n*aden)
#     b = bnum*1./(n*bden)
#     r = 2./(N*(N-1))*aden
#     return (a,b,r)
#     
# def q_step(n,a,b,r,E):
#     """Run the maximization/q step of the EM algorithm.
#     
#     n - number of times that each edge has been measured. Integer
#     a - current estimate for a. Float.
#     b - current estimate for b. Float.
#     r - current estimate for rho. Float.
#     E - actual observed number of edges. Numpy array.
#     """
#     N = E.shape[0]
#     q = np.zeros(E.shape)
#     for j in range(1,N):
#         for i in range(j):
#             p1 = r*(a**E[i,j])*(1-a)**(n-E[i,j])
#             p2 = (1-r)*(b**E[i,j])*(1-b)**(n-E[i,j])
#             q[i,j] = p1*1./(p1+p2)
#     return (q)
#     
#     
#variable n
def abr_step(N,E,Q):
    """Run the expectation (abr) step of the EM algorithm.
    This variant assumes that the true-positive and false-postive rates are global.
    
    N - number of times that each edge has been measured. Numpy array.
    E - actual observed number of edges in numpy array form.
    Q - current estimate than edge is actually present. Numpy array.
    """
    # Step 0: establish variables
    n = E.shape[0] #the number of nodes in the network, following Newman's notation

    
    #Temporary variables to hold the numerator and denominator sums of the alpha, 
    ## beta. and rho estimates.
    anum = 0
    aden = 0
    bnum = 0
    bden = 0
    rnum = 0

    #Step 1: loop through the upper triangle of matrix Q to calculate the sums
    for j in range(1,n):
        for i in range(j):
            anum += E[i,j]*Q[i,j]
            bnum += E[i,j]*(1-Q[i,j])
            rnum += Q[i,j]
            aden += N[i,j]*Q[i,j]
            bden += N[i,j]*(1-Q[i,j])
            
    # Step 2: calculate alpha, beta, and rho
    alpha = anum*1./(aden)
    beta = bnum*1./(bden)
    rho = 2./(n*(n-1))*rnum
    
    # Step 3: return alpha, beta, and rho
    return (alpha,beta,rho)
    
def q_step(N,E,alpha,beta,rho):
    """Run the maximization/q step of the EM algorithm.
    This variant assumes that the true-positive and false-postive rates are global.
    
    N - number of times that each edge has been measured. Numpy array.
    E - actual observed number of edges. Numpy array.
    alpha - current estimate for alpha, true positive rate. Float.
    beta - current estimate for beta, false positive rate. Float.
    rho - current estimate for rho, density of network. Float.
    """
    # Step 0: establish variables
    n = E.shape[0] #the number of nodes in the network, following Newman's notation

    
    # Create an array to store q estimates
    Q = np.zeros(E.shape)
    
    #Step 1: loop through the upper triangle of the network to calculate new Q
    for j in range(1,n):
        for i in range(j):
            p1 = rho*(alpha**E[i,j])*(1-alpha)**(N[i,j]-E[i,j])
            p2 = (1-rho)*(beta**E[i,j])*(1-beta)**(N[i,j]-E[i,j])
            Q[i,j] = p1*1./(p1+p2)
    #Step 2: return Q
    return (Q)

def richnoisy(N,E,tolerance = .000001,seed=10):
    """Run the Expectation-Maximization (EM) algorithm proposed in Newman 2018.
    This algorithm takes as an input a matrix E of times that an edge was observed
    to exist, as well as a matrix (or integer) N of times that an edge was observed
    in general.
    
    This algorithm assumes that true-positive and false-positive rates are global,
    rather than local, properties. 
    
    N - number of times that each edge has been measured. Numpy array or integer.
    E - number of times that each edge has been actually observed. Numpy array.
    tolerance - when to stop iterating the algorithm. Float.
    
    Returns:
        alpha - estimate of true-positive rate.
        beta - estimate of false-positive rate.
        rho - estimate of network density.
        Q - estimates of edge existence probability.
    
    For more information, please consult:
    
    Newman, M.E.J. 2018. “Network structure from rich but noisy data.” Nature Physics 14 6 (June 1): 542–545. doi:10.1038/s41567-018-0076-1.
    """
    # Step 0: establish variables
    
    # If N is an integer, create a matrix to use for these calculations
    if np.size(N) == 1:
        N = np.ones((n,n))*N
    
    # Record iterations and previous values to confirm convergence
    iterations = 0
    alpha_prev = 0
    beta_prev = 0
    rho_prev = 0    
    
    # Step 1: Do an initial q-step with random alpha, beta, and rho values
    # Randomly assign values to alpha, beta, and rho to start
    np.random.seed(seed)
    beta, alpha, rho = np.sort(np.random.rand(3)) #beta must be smaller than alpha
    ## as 
    
    #Now calculate initial Q
    Q = q_step(N, E, alpha, beta, rho)
    
    # Step 2: Repeat until tolerance is met
    while abs(alpha_prev-alpha) > tolerance or abs(beta_prev-beta) > tolerance or abs(rho_prev - rho) > tolerance:
        alpha_prev = alpha
        beta_prev = beta
        rho_prev = rho
        alpha, beta, rho = abr_step(N, E, Q)
        Q = q_step(N, E, alpha, beta, rho)
        iterations += 1
    
    # Step 3: return values
    return (alpha, beta, rho, Q, iterations)    
    
def falsediscovery(alpha,beta,rho):
    return (1-rho)*beta/(rho*alpha + (1-rho)*beta)
    
    

# #variable n complete, but rewrite this to assume that each edge has a different a and b
# def abr_step_varn(n,E,Q):
#     """Run the expectation (abr) step of the EM algorithm.
#     
#     n - number of times that each edge has been measured. Numpy array.
#     E - actual observed number of edges in numpy array form.
#     Q - current estimate than edge is actually present. Numpy array.
#     """
#     # Get N
#     N = E.shape[0]
#     
#     #Calculate a,b, and r
#     anum = 0
#     aden = 0
#     bnum = 0
#     bden = 0
#     rnum = 0
#     r = 0
#     for j in range(1,N):
#         for i in range(j):
#             anum += E[i,j]*Q[i,j]
#             aden += n[i,j]*Q[i,j]
#             bnum += E[i,j]*(1-Q[i,j])
#             bden += n[i,j]*(1-Q[i,j])
#             rnum += Q[i,j]
#     a = anum*1./(aden)
#     b = bnum*1./(bden)
#     r = 2./(N*(N-1))*rnum
#     return (a,b,r)
#     
# def q_step_varn(n,E,a,b,r):
#     """Run the maximization/q step of the EM algorithm.
#     
#     n - number of times that each edge has been measured. Integer
#     a - current estimate for a. Float.
#     b - current estimate for b. Float.
#     r - current estimate for rho. Float.
#     E - actual observed number of edges. Numpy array.
#     """
#     N = E.shape[0]
#     q = np.zeros(E.shape)
#     for j in range(1,N):
#         for i in range(j):
#             p1 = r*(a**E[i,j])*(1-a)**(n[i,j]-E[i,j])
#             p2 = (1-r)*(b**E[i,j])*(1-b)**(n[i,j]-E[i,j])
#             q[i,j] = p1*1./(p1+p2)
#     return (q)

if __name__ == '__main__':
    test = np.genfromtxt('documents\\projects\\code\\test_RichNoisy.csv',delimiter=',')
    test[0,0] = 0

    a, b, r, Q = richnoisy(n_test,test)
        
        
    test = np.genfromtxt('C:\\\\users\\cabaniss\\documents\\projects\\code\\test_RichNoisy.csv',delimiter=',')
    test[0,0] = 0
    n_test = np.array([[.5,5,3,5,10],[0,.5,5,1,5],[0,0,.5,5,5],[0,0,0,.5,5],[0,0,0,0,.5]])
    n_test =  n_test + n_test.T
        
    n = 5    
    alist = [.75]
    blist = [.25]
    rlist = [1./test.shape[0]]
    
    q = q_step_varn(n_test,alist[0],blist[0],rlist[0],test)
    
    for x in range(10):
        a, b, r = abr_step_varn(n_test,test,q)
        alist.append(a)
        blist.append(b)
        rlist.append(r)
        q = q_step_varn(n_test,a,b,r,test)
        
    plot(np.triu(test).flatten(),q.flatten(),'bo')    
    plot(np.triu(test).flatten()*1./np.triu(n_test).flatten(),q.flatten(),'bo')