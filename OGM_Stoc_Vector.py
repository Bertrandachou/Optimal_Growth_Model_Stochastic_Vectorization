from __future__ import division

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:07:54 2013

@author: Bertrand Achou
"""

""" This is the stochastic optimal growth model

we make use of interpolation
with 20 grids for value function and 1000 points for next period capital 
the algorithm converges in about 19 sec

"""

import numpy as np
from scipy.interpolate import interp1d

"""first we create a dictionary containing all our parameters of the model"""

p = {'alpha': 0.3, 'beta': 0.95, 'sigma': 1, 'delta': 0.3 }


"""create a dictionary for the shocks"""

shock = { 'A': np.array([0.85,0.9,0.95,1,1.05,1.1,1.15]) , 'transit': np.array([[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1]]) } 



kss = ( (1 - p['beta'] * (1 - p['delta']) ) / (p['alpha'] * p['beta']) )**( 1 / (p['alpha'] - 1))
sd  = 0.9


pgrid = { 'n': 20,'n1':1000, 'kmin': kss*(1-sd), 'kmax': kss*(1+sd) }

# n is the number of point in the kgrid
# n1 is the number of points in the kpgrid 

Agrid = np.array([[[shock['A'][i] for j in xrange(pgrid['n'])] for k in xrange(pgrid['n1'])] for i in xrange(len(shock['A']))])

kgrid    = np.array([[np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n']) for i in xrange(pgrid['n1'])] for k in xrange(len(shock['A']))])

kpgrid1    = np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n1'])
kpgrid11   = np.array([[kpgrid1[i] for j in xrange(pgrid['n'])] for i in xrange(pgrid['n1']) ] )
kpgrid     = np.array( [[[kpgrid1[i] for j in xrange(pgrid['n'])] for i in xrange(pgrid['n1'])] for l in xrange(len(shock['A']))] )


# we build our guess value function
# first line of the matrix corresponds to the first level of productivity
# first column corresponds to the first level of capital 
    
V0 = np.array([ [ np.zeros(pgrid['n']) for k in xrange(pgrid['n1']) ] for j in xrange(len(shock['A']))] )   


""" we then define the functions of the model"""

def utility(c,p):
    # this is the utility function 
    # c is consumption
    # p is the dictionary of parameters
    if p['sigma'] == 1:
        return np.log(c)
    else:
        return ( c**(1 - p['sigma']) ) / ( 1 - p['sigma'] )
        

def production(k,p,A):
    # this is the production function of the model
    # k is capital
    # p is a dictionary of parameters
    return A * k**(p['alpha'])
    
    
def new_value(k,kp,kpp,V0,p,pgrid,A,shock):
    # computes the new value function for k
    # given the matrix kp which is similar to k 
    # except that values are ordered in column (k is ordered in rows and is square)
    # knowing that the former guess on the value function was V0
    # and for a given dictionnary of parameters p
    
    # kpp is kpgrid11 in our program
    # it is a grid for second period capital as kp
    
    # we use Boolean indexing checking the values kprime
    # that satisfy the constraint
    # when the resource constraint is not satisfied we impose a large penalty 
    # to the representative agent to ensure that he will never choose these 
    # values
    
    
    new_V0 = np.array([ [ np.zeros(pgrid['n']) for z in xrange(pgrid['n1']) ] for j in xrange(len(shock['A']))] )   

    budget_not           = ((production(k,p,A) + (1 - p['delta']) * k - kp) < 0)
    
    ctemp               = production(k,p,A) + (1 - p['delta']) * k - kp
    
    ctemp[budget_not]  = 0.001  
   
    utemp               = utility(ctemp,p)
    
    f                   = interp1d(k[0,0],V0[:,0],kind ='quadratic')
    
    VP0                 = f(kpp)
      
    for i in xrange(len(shock['A'])):
        
        for j in xrange(len(shock['A'])):
            
            utemp[i] += p['beta'] *  shock['transit'][i,j] * VP0[j]
     
    utemp[budget_not] = -99999999
    
    Vtemp        = utemp.reshape(len(shock['A']),pgrid['n']*pgrid['n1'])
    

    for i in xrange(len(shock['A'])):
                
        for z in xrange(pgrid['n']):
            
            new_V0[i,:,z] = max(Vtemp[i][z::pgrid['n']])
            
             
    return new_V0
    


""" Now we run the algorithm """

crit    = 100
epsilon = 10**(-6)

oiter = 0 

while crit > epsilon:
    
    
    TV  = new_value(kgrid,kpgrid,kpgrid11,V0,p,pgrid,Agrid,shock)
    
    critmat      = abs(V0[:,0,:]-TV[:,0,:]).reshape(1,pgrid['n']*len(shock['A']))
    crit         = max(critmat[0])
    
    
    V0        = TV



  
"""import pylab

kgrid2 = np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n'])
   
pylab.plot(kgrid2,V0[0,0,:])
pylab.plot(kgrid2,V0[1,0,:])
pylab.plot(kgrid2,V0[2,0,:])
pylab.plot(kgrid2,V0[3,0,:])
pylab.plot(kgrid2,V0[4,0,:])
pylab.plot(kgrid2,V0[5,0,:])
pylab.plot(kgrid2,V0[6,0,:])

"""