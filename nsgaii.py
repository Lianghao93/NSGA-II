import numpy as np
import nd_sort
import crowding_distance
import tournament
import variation


class Global:
    def __init__(self, d=10, N=100, M=2):
        self.d = d
        self.N = N
        self.M = M
        self.upper = np.ones((1, d))
        self.lower = np.zeros((1, d))


def cost_fun(x):
    obj = []
    d = 10
    a1 = np.random.randn(1, d)
    a2 = np.random.randn(1, d)

    for i in range(len(x)):
        f1 = float(np.dot(a1, np.power(x[i][:], 2).transpose()))
        f2 = float(np.dot(a2, -np.power(x[i][:], 3).transpose()))
        obj.append([f1, f2])
    return np.array(obj)

def INDIVIDUAL(decs):
    pop_obj = cost_fun(decs)
    return [decs, pop_obj]


class nsgaii(object):
    def __init__(self, decs=None, ite=100, eva=100*100, opt=Global):
        self.decs = decs
        self.ite = ite
        self.eva = eva
        self.opt = opt

    def initialize(self):
        pop_dec = np.random.rand(self.opt.N, self.opt.d) * (self.opt.upper - self.opt.lower) + self.opt.lower
        pop_obj = cost_fun(pop_dec)
        return [pop_dec, pop_obj]

    def run(self):
        if self.decs is None:
            population = self.initialize()
        else:
            population = INDIVIDUAL(self.decs)

        front_no, max_front = nd_sort(population[1], np.inf)
        crowd_dis = crowding_distance(population[1], front_no)
        for i in range(self.ite):
            mating_pool = tournament(2,Global.N,front_no, crowd_dis)
            offspring = variation(population[mating_pool][:])
            population = population.append(offspring)





