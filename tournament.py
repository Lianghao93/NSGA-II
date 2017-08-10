#!/usr/bin/env python
# encoding: utf-8
import numpy as np

def tournament(K, N, fit):
    '''
    tournament selection
    :param K: number of solutions to be compared
    :param N: number of solutions to be selected
    :param fit: fitness vectors
    :return: index of selected solutions
    '''
    n = len(fit)
    mate = []
    for i in range(N):
        a = np.random.randint(n)
        for j in range(K):
            b = np.random.randint(n)
            for r in range(fit[(0, :)].size):
                if fit[(b, r)] < fit[(a, r)]:
                    a = b
        mate.append(a)
    
    return np.array(mate)
