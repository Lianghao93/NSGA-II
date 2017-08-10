import numpy as np
from nd_sort import nd_sort
fron crowding_distance import crowding_distance
from tournament import tournament
from environment_selection import environment_selection
from GLOBAL import Global



Global = Global(lower=-np.ones((1,10)),upper=np.ones((1,10)))

class nsgaii(object):
    def __init__(self, decs=None, ite=100, eva=100*100):
        self.decs = decs
        self.ite = ite
        self.eva = eva

    def run(self):
        if self.decs is None:
            population = Global.initialize()
        else:
            population = Global.individual(self.decs)

        front_no, max_front = nd_sort(population[1], np.inf)
        crowd_dis = crowding_distance(population[1], front_no)
        while self.eva>=0:
            mating_pool = tournament(2,Global.N,front_no, crowd_dis)
            parent = [population[0][mating_pool], population[1][mating_pool]]
            offspring = Global.variation(parent[0])
            population = Global.unit_population(population, offspring)
            population = environment_selection(population,Global.N)
        return population
   




