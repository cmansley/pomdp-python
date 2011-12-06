import math
import random

import itertools

from cvxopt import matrix 
from cvxopt import solvers

solvers.options['show_progress'] = False

#
# Auxiliary Functions
#
from itertools import izip, imap, count
def argmax(values, key=None):
    """
    """
    if key:
        return max(izip(values, count()), key=lambda x: key(x[0]))
    else:
        return max(izip(values, count()))

from itertools import imap
import operator
def dotproduct(vec1, vec2):
    """
    """
    return sum(imap(operator.mul, vec1, vec2))

# Optimized with local variables
# def dotproduct(vec1, vec2, sum=sum, imap=imap, mul=operator.mul):
#    return sum(imap(mul, vec1, vec2))

# Non-optimized silliness
# def dot(x,y):
#    return sum([x[i]*y[i] for i in range(len(x))])


#
# Model Specific Functions
#
def tau(state, action, nextstate, alpha):
    pass

# def tau(action, observation, alpha):
#    "POMDPs"
#    pass


#
# Incremental Pruning Functions
#

def crosssum(A,B):
    """
    Input: A and B are lists of vectors (matrices)
    Output: A set of vectors that is all combinations of sums from the two sets.
    """
    return [x+y for x,y in itertools.product(A,B)]


def filter(F):
    wi = set()
    fi = set(range(len(F)))
    for phi in F:
        _, pos = argmax(phi, lambda x: x[i])
        wi.add(pos)
        fi.discard(pos)

    while fi:
        W = [F[x] for x in wi]
        i = fi.pop()
        phi = F[i]
        x = dominate(phi, W)
        
        if x:
            fi.add(i)
            _, pos = argmax([F[x] for x in fi], lambda y: dot(x,y))
            wi.add(pos)
            fi.discard(pos)
            
    return [F[x] for x in wi]
        
        

def dominate(alpha, setA):
    """
    Input: alpha is a vector in matrix form 
           setA is a list of vectors, not including alpha
    Output: none if there is no solution
            a vector for information state, otherwise
    """
    
    # Original Linear Program
    # max delta
    # s.t.
    # x alpha >= delta + x alpha_p for all alpha_p in A\{alpha}
    # sum(x_i) == 1
    # all x_i > 0

    # Cannonical Form 
    # max delta
    # s.t.
    # delta + x (alpha_p - alpha) <= 0 for all alpha_p in A\{alpha}
    # -Ix <= 0
    #
    # e'x == 1

    # where e is the vector of all ones

    # construct c
    t1 = matrix(0.0 , (len(alpha), 1))
    t2 = matrix([1.0])
    c = -1*matrix([t1,t2])

    # inequalities

    # construct A
    # idenity matrix for -Ix <= 0
    t3 = matrix(0.0, (len(alpha), len(alpha)+1))
    t3[::len(alpha)+1] = -1
    A = t3
    
    # x (alpha_p - alpha) + delta <= 0
    for vector in setA:
        da = vector - alpha
        t4 = matrix([1.0])
        t5 = matrix([da, t4])
        A = matrix([A,t5.T])

    # construct b
    b = matrix(0.0, (len(alpha)+len(setA), 1))

    # equalities

    # construct G
    t6 = matrix(1.0, (len(alpha),1))
    t7 = matrix([0.0])
    t8 = matrix([t6, t7])
    G = t8.T
    
    # construct h
    h = matrix([1.0])

    # solve!
    sol=solvers.lp(c,A,b,G,h)

    # fail on infeasible solution
    if sol['status'].find('optimal') == -1:
        return None

    delta = sol['x'][len(alpha)]
    if delta > 0:
        # return state
        return sol['x'][0:len(alpha), 0]

    return None


if __name__ == "__main__":
    alpha = matrix([0.2,0.2])
    
    t1 = matrix([0.1,0.4])

    A = []
    A.append(t1)

    print dominate(alpha, A)
