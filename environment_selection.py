#!/usr/bin/env python
# encoding: utf-8
import numpy as np
from nd_sort import nd_sort
from crowding_distance import crowding_distance


def environment_selection(population, N):
    '''
    environmental selection in NSGA-II
    :param population: current population
    :param N: number of selected individuals
    :return: next generation population
    '''
    front_no, max_front = nd_sort(population[1], N)
    next_label = [False for i in range(front_no.size)]
    for i in range(front_no.size):
        if front_no[i] < max_front:
            next_label[i] = True
    crowd_dis = crowding_distance(population[1], front_no)
    last = [i for i in range(len(front_no)) if front_no[i]==max_front]
    rank = np.argsort(-crowd_dis[last])
    delta_n = rank[: (N - int(np.sum(next_label)))]
    rest = [last[i] for i in delta_n]
    for i in rest:
        next_label[i] = True
    index = np.array([i for i in range(len(next_label)) if next_label[i]])
    next_pop = [population[0][index,:], population[1][index,:]]
    return next_pop, front_no[index], crowd_dis[index],index
