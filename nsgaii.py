import numpy as np
from nd_sort import nd_sort
fron crowding_distance import crowding_distance
from tournament import tournament
from environment_selection import environment_selection
from GLOBAL import Global

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
            fit = np.vstack((front_no, crowd_dis)).T
            mating_pool = tournament(2 ,Global.N, fit)
            parent = [population[0][mating_pool], population[1][mating_pool]]
            offspring = Global.variation(parent[0])
            population = Global.unit_population(population, offspring)
            population = environment_selection(population, Global.N)
        return population
    
    def draw(self):
        population = self.run()
        pop_obj = population[1]
        front_no, max_front = nd_sort(pop_obj, 1)
        non_dominated = pop_obj[front_no==1, :]
        if Global.M == 2:
            plt.scatter(non_dominated[0, :], non_dominated[1, :])
        elif Global.M == 3:
            x,y,z = non_dominated[0, :], non_dominated[1, :],non_dominated[2, :]
            ax = plt.subplot(111,projection='3d') 
            ax.scatter(x, y,z,c='b') 
        else:
            for i in range(len(non_dominated)):
                plt.plot(range(1, Global.M + 1), non_dominated[i, :])
         
   




