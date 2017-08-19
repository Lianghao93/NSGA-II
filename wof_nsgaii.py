
#!/usr/bin/env python
# encoding: utf-8
import math
import numpy as np
from nd_sort import nd_sort
from crowding_distance import crowding_distance
from tournament import tournament
from environment_selection import environment_selection
import time
from GLOBAL import Global

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Global = Global(M=2, d=1000,lower=-np.ones((1, 1000)), upper=np.ones((1, 1000)))


class wof_nsgaii(object):
    """
    nsgaii for large-scale multi-objective optimization problems
    """

    def __init__(self, delta=0.5, group_no=4, p=0.2, choose_no=3, weight_size=10, eva=100 * 100, decs=None, ite=100):
        """

        :param delta: delta
        :param group_no: number of groups
        :param p: p value
        :param choose_no: number of chosen solutions
        :param weight_size: number of weight vectors
        """
        self.ite = ite
        self.decs = decs
        self.eva = eva
        self.t1 = int(0.01 * self.eva)
        self.t2 = int(0.5 * self.t1)
        self.delta = delta
        self.group_no = group_no
        self.choose_no = choose_no
        self.p = p
        self.weight_size = weight_size

    def run(self):
        """
        run the wof_nsgaii to obtain the final population
        :return: the final population
        """
        start = time.clock()
        if self.decs is None:
            population = Global.initialize()
        else:
            population = Global.individual(self.decs)

        g_size = math.ceil(Global.d / self.group_no)
        group = [[] for i in range(self.group_no)]
        lower = np.tile(Global.lower, (Global.N, 1))
        upper = np.tile(Global.upper, (Global.N, 1))
        w_lower = np.zeros((1,self.group_no))
        w_upper = np.ones((1, self.group_no))*2
        evaluated = 0
        evaluated = evaluated + Global.N
        front_no, max_front = nd_sort(population[1], np.inf)
        crowd_dis = crowding_distance(population[1], front_no)
        while evaluated <= self.eva:
            en1 = evaluated
            while evaluated < (self.t1 + en1):
                fit = np.vstack((front_no, -crowd_dis)).T
                mating_pool = tournament(2, Global.N, fit)
                pop_dec, pop_obj = population[0], population[1]
                parent = [pop_dec[mating_pool, :], pop_obj[mating_pool, :]]
                offspring = Global.individual(Global.variation(parent[0]))
                evaluated = evaluated + Global.N
                population = [np.vstack((population[0], offspring[0])),np.vstack((population[1], offspring[1]))]
                population, front_no, crowd_dis = environment_selection(population, Global.N)
            crowd_index = np.argsort(-crowd_dis)
            pop_dec, pop_obj = population[0], population[1]
            x_q = [pop_dec[crowd_index[:self.choose_no], :], pop_obj[crowd_index[:self.choose_no], :]]
            k = 1
            sq = [np.tile(pop_dec, (self.choose_no, 1)), np.tile(pop_obj, (self.choose_no, 1))]
            while k <= self.choose_no:
                x_dec = np.zeros((self.weight_size, Global.d))
                temp = x_q[0][k - 1, :]
                g_sort = np.argsort(temp)
                w = 2 * np.random.random((self.weight_size, self.group_no))
                for i in range(self.group_no):
                    group[i] = g_sort[int(g_size * i):int(g_size * (i + 1))]
                    x_dec[:, group[i]] = np.tile(temp[group[i]], (self.weight_size, 1)) \
                                         + self.p * (np.tile(w[:,i].reshape(len(w[:,i]),1), (1, g_size)) - 1) \
                                           * np.tile(Global.upper[:,group[i]] - Global.lower[:,group[i]], (self.weight_size, 1))
                x_dec = np.maximum(np.minimum(x_dec, Global.upper), Global.lower)
                en2 = evaluated
                X = Global.individual(x_dec)
                evaluated = evaluated + x_dec.shape[0]
                w_front,_ = nd_sort(X[1], np.inf)
                w_crowd = crowding_distance(X[1], w_front)
                while evaluated < (self.t2 + en2):
                    fit = np.vstack((w_front, -w_crowd)).T
                    w_mating_pool = tournament(2, self.weight_size, fit)
                    w_offspring = Global.variation(w[w_mating_pool, :self.group_no], boundary=[w_lower,w_upper])
                    for i in range(self.group_no):
                        x_dec[:, group[i]] = np.tile(temp[group[i]], (self.weight_size, 1)) \
                                             * np.tile(w_offspring[:, i], (1, g_size))
                    x_dec = np.maximum(np.minimum(x_dec, upper), lower)
                    x_offspring = Global.individual(x_dec)
                    evaluated = evaluated + x_dec.shape[0]
                    w, front_no, crowd_dis = environment_selection([np.vstack((population[0], x_offspring[0])),np.vstack((population[1], x_offspring[1]))],
                                                                   self.weight_size)
                s_dec = population[0].copy()
                for i in range(self.group_no):
                    s_dec[:, group[i]] = population[0][:, group[i]] + \
                                         self.p * (np.tile(w[math.ceil(self.weight_size / 2), i], (Global.N, g_size)) - 1) \
                                         * np.tile(upper[group[i]] - lower[group[i]], (Global.N, 1))
                s_dec = np.maximum(np.minimum(s_dec, upper), lower)
                S = Global.individual(s_dec)
                evaluated = evaluated + s_dec.shape[0]
                sq[0][Global.N * (k - 1):Global.N * k, :] = S[0]
                sq[1][Global.N * (k - 1):Global.N * k, :] = S[1]
                k += 1
            sq[0][Global.N * self.choose_no:, :] = population[0]
            sq[1][Global.N * self.choose_no:, :] = population[1]
            population, front_no, crowd_dis = environment_selection(sq, Global.N)
        while evaluated < self.eva:
            fit = np.vstack((front_no, -crowd_dis)).T
            mating_pool = tournament(2, Global.N, fit)
            pop_dec, pop_obj = population[0], population[1]
            parent = [pop_dec[mating_pool, :], pop_obj[mating_pool, :]]
            offspring = Global.variation(parent[0])
            evaluated = evaluated + Global.N
            population = [np.vstack((population[0], offspring[0])),np.vstack((population[1], offspring[1]))]
            population, front_no, crowd_dis = environment_selection(population, Global.N)
        return population

    def draw(self):
        population = self.run()
        pop_obj = population[1]
        front_no, max_front = nd_sort(pop_obj, 1)
        non_dominated = pop_obj[front_no == 1, :]
        if Global.M == 2:
            plt.scatter(non_dominated[0, :], non_dominated[1, :])
        elif Global.M == 3:
            x, y, z = non_dominated[:, 0], non_dominated[:, 1], non_dominated[:, 2]
            ax = plt.subplot(111, projection='3d')
            ax.scatter(x, y, z, c='b')
        else:
            for i in range(len(non_dominated)):
                plt.plot(range(1, Global.M + 1), non_dominated[i, :])


a = wof_nsgaii()
b = a.draw()
plt.show()
