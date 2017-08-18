#!/usr/bin/env python
# encoding: utf-8
import numpy as np

class Global(object):
    """
    The problem related parameters and genetic operations
    """
    def __init__(self, d=10, n=100, M=2, lower=-np.ones((1, 10)), upper=np.ones((1, 10))):
        self.d = d
        self.N = n
        self.M = M
        self.upper = upper
        self.lower = lower

    def cost_fun(self, x):
        """
        calculate the objective vectors
        :param x: the decision vectors
        :return: the objective vectors
        """
        n = x.shape[0]
        a = np.zeros((self.M, self.d))
        for i in range(self.d):
            for j in range(self.M):
                a[j,i] = ((i+0.5)**(j-0.5))/(i+j+1.)
        obj = np.zeros((n, self.M))
        for i in range(n):
            for j in range(self.M):
                obj[i, j] = np.dot(x[i, :] ** (j + 1), a[j, :].T)
        return obj

    def individual(self, decs):
        """
        turn decision vectors into individuals
        :param decs: decision vectors
        :return: individuals
        """
        pop_obj = self.cost_fun(decs)
        return [decs, pop_obj]

    @staticmethod
    def unit_population(population, offspring):
        """
        Combine two population
        :param population: parent population
        :param offspring: offspring population
        :return:
        """
        return [
            np.vstack((population[0], offspring[0])),
            np.vstack((population[1], offspring[1]))]

    @property
    def initialize(self):
        """
        initialize the population
        :return: the initial population
        """
        pop_dec = np.random.random((self.N, self.d)) * (self.upper - self.lower) + self.lower
        return self.individual(pop_dec)

    def variation(self, pop_dec):
        """
        Generate offspring individuals
        :param pop_dec: decision vectors
        :return:
        """
        pro_c = 1
        dis_c = 20
        pro_m = 1
        dis_m = 20
        pop_dec = pop_dec[:(len(pop_dec) // 2) * 2][:]
        (n, d) = np.shape(pop_dec)
        parent_1_dec = pop_dec[:n // 2, :]
        parent_2_dec = pop_dec[n // 2:, :]
        beta = np.zeros((n // 2, d))
        mu = np.zeros((n // 2, d))
        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (dis_c + 1))
        beta[mu > 0.5] = np.power(2 * mu[mu > 0.5], -1 / (dis_c + 1))
        beta = beta * -1 * np.random.randint(2, size=(n // 2, d))
        beta[np.random.random((n // 2, d)) < 0.5] = 1
        beta[np.tile(np.random.random((n // 2, 1)) > pro_c, (1, d))] = 1
        offspring_dec = np.vstack(((parent_1_dec + parent_2_dec) / 2 + beta * (parent_1_dec - parent_2_dec) / 2,
                                   (parent_1_dec + parent_2_dec) / 2 - beta * (parent_1_dec - parent_2_dec) / 2))
        site = np.random.random((n, d)) < pro_m / d
        mu = np.random.random((n, d))
        temp = site & (mu <= 0.5)
        lower = np.tile(self.lower, (n, 1))
        upper = np.tile(self.upper, (n, 1))
        norm = (offspring_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
        offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                               (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1. - norm, dis_m + 1.),
                                         1. / (dis_m + 1)) - 1.)
        temp = site & (mu > 0.5)
        norm = (upper[temp] - offspring_dec[temp]) / (upper[temp] - lower[temp])
        offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                               (1. - np.power(
                                   2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(1. - norm, dis_m + 1.),
                                   1. / (dis_m + 1.)))
        offspring_dec = np.maximum(np.minimum(offspring_dec, upper), lower)
        return self.individual(offspring_dec)
