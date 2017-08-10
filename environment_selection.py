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
    next_label = front_no < max_front
    crowd_dis = crowding_distance(population[1], front_no)
    last = np.array([i for i in range(len(front_no)) if front_no[i]==max_front])
    rank = np.argsort(-crowd_dis[last])
    delta_n = rank[: (N - next_label.sum())]
    next_label[last[delta_n]] = True
    next_pop = [population[0][next_label], population[1][next_label]]
    return next_pop, front_no, crowd_dis
