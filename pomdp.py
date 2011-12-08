from __future__ import division

import math
import random

import itertools

from collections import defaultdict

from cvxopt import matrix 
from cvxopt import solvers
from cvxopt import mul

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
# Model Specific Class
#
class Model:
    def __init__(self, S, A, O, R, T, M, gamma):
        self.S = S
        self.A = A
        self.O = O
        self.R = R
        self.T = T
        self.M = M
        self.gamma = gamma

    def tau(self, action, observation, S):
        """POMDPs"""
        ts = []
        for alpha in S:
            t1 = mul(alpha, self.M[(observation,action)])
            t2 = self.T[action] * t1
            t3 = 1/len(self.O) * self.R[action] + self.gamma*t2
            ts.append(t3)

        return ts

    # def tau(state, action, nextstate, S):
    #    """HM-MDPs"""
    #    pass


#
# Incremental Pruning Functions
#
def union(Sa):
    Su = []
    
    for s in Sa.values():
        for v in s:
            diff = 0
            total = 0
            for u in Su:
                total += 1
                t = v-u
                if sum(t*t.T) > 0.01:
                    diff += 1

            if diff == total:
                Su.append(v)

    return Su

def unionify(S):
    """
    Input: S is a list of vectors
    Output: A list of vectors that are more like a union (very similar vectors are removed)
    """
    St = []

    for v in S:
        diff = 0
        total = 0
        for u in St:
            total += 1
            t = v-u
            if sum(t*t.T) > 0.01:
                diff += 1

        if diff == total:
            St.append(v)

    return St

def vi(pomdp):
    S = []
    #for a in pomdp.A:
    #    S.append(R[a])
    S.append(matrix([1,1,1,1]))

    for x in range(400):
        Saz = {}
        Sa = {}
        for action in pomdp.A:
            for obs in pomdp.O:
                Saz[obs] = filter(pomdp.tau(action, obs, S))
            Sa[action] = incprune(Saz)
                        
        S = filter(union(Sa))
        print x, len(S)

    for a in pomdp.A:
        for v in Sa[a]:
            print a, v

    return Sa


def incprune(Saz):
    vects = Saz.values()
    W = filter(crosssum(vects[0],vects[1]))
    for i in range(2, len(vects)):
        W = filter(crosssum(W,vects[i]))

    return W


def crosssum(A,B):
    """
    Input: A and B are lists of vectors (matrices)
    Output: A set of vectors that is all combinations of sums from the two sets.
    """
    return [x+y for x,y in itertools.product(A,B)]


def filter(F):
    """
    Input: F is a list of vectors (matrix)
    Output: A reduced list of vectors uniquely identifying the value function.
    """
    F = unionify(F)
    wi = set()
    fi = set(range(len(F)))
    for i in range(4): # fix me!!!
        _, pos = argmax(F, lambda x: x[i])
        wi.add(pos)
        fi.discard(pos)

    while fi:
        W = [F[ii] for ii in wi]
        i = fi.pop()
        phi = F[i]
        x = dominate(phi, W)

        if x:
            fi.add(i)
            idx = [ii for ii in fi]
            _, pos = argmax([F[ii] for ii in fi], lambda y: dotproduct(x,y))
            wi.add(idx[pos])
            fi.discard(idx[pos])
            
    return [F[ii] for ii in wi]
        
        

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


#
# Test Suite
#
if __name__ == "__main__":

    #
    # Littman 1D
    #

    S = ['left', 'middle', 'right', 'goal']
    A = ['w0', 'e0']
    O = ['nothing', 'goal']

    R = {}
    R['w0'] = matrix([0, 0, 0, 1])
    R['e0'] = matrix([0, 0, 0, 1])

    # transitions P(s'|s,a) or T[(s',a)] = [s1,s2, ...]
    T = {}
    T['w0'] = matrix([[1.0, 0.0, 0.0, 0.0],[1.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0],[0.333333, 0.333333, 0.333333, 0.0]]).T
    T['e0'] = matrix([[0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 1.0, 0.0],[0.333333, 0.333333, 0.333333, 0.0]]).T

    # observation emmissions P(z|s,a) or M[(z,a)] = [s1,s2 ..]
    M = {}
    M[('nothing','w0')] = matrix([1.0, 1.0, 1.0, 0.0])
    M[('goal','w0')] = matrix([0.0, 0.0, 0.0, 1.0])

    M[('nothing','e0')] = matrix([1.0, 1.0, 1.0, 0.0])
    M[('goal','e0')] = matrix([0.0, 0.0, 0.0, 1.0])

    pomdp = Model(S,A,O,R,T,M, 0.75)

    Sa = vi(pomdp)


    tests = []
    tests.append(matrix([0.5,0.0,0.0,0.5]))
    tests.append(matrix([0.5,0.0,0.1,0.4]))
    tests.append(matrix([0.25,0.25,0.25,0.25]))
    tests.append(matrix([0.3,0.1,0.4,0.2]))
    tests.append(matrix([0.2,0.2,0.3,0.3]))

    # arc solver
    support = matrix([0.9931026876124613433916011, 0.9931026876124613433916011, 1.7655158894202862551736644, 1.0206878527024387803834315])

    # chris solver
    for x in tests:
        mx = 0
        for v in Sa['w0']:
            val = sum(x.T*v)
            if val > mx:
                mx = val
        print mx - sum(x.T*support)
