#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def nd_sort(pop_obj, n_sort):
    '''
    :param n_sort:
    :param pop_obj: objective vectors
    :return: [FrontNo, MaxFNo]
    '''
    n, m_obj = np.shape(pop_obj)
    a, loc = np.unique(pop_obj[:, 0], return_inverse = True)
    index = pop_obj[:, 0].argsort()
    new_obj = pop_obj[index, :]
    front_no = np.inf * np.ones(n)
    max_f_no = 0
    while np.sum(front_no < np.inf) < min(n_sort, len(loc)):
        max_f_no += 1
        for i in range(n):
            if front_no[i] == np.inf:
                dominated = False
                for j in range(i, 0, -1):
                    if front_no[j-1] == max_f_no:
                        m = 2
                        while m <= m_obj and new_obj[i][m - 1] >= new_obj[j][m - 1]:
                            m += 1
                        dominated = m > m_obj
                        if not dominated or m_obj ==2:
                            break
                if not dominated:
                    front_no[i] = max_f_no
    return front_no[loc], max_f_no
