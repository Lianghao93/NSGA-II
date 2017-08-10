class Global(object):
    
    def __init__(self, d, n, M, upper, lower):
        self.d = d
        self.N = n
        self.M = M
        self.upper = upper
        self.lower = lower

    
    def cost_fun(self, x):
        '''
        calculate the objective vectors
        :param x: the decision vectors
        :return: the objective vectors
        '''
        a1 = np.random.randn(1, self.d)
        a2 = np.random.randn(1, self.d)
        f1 = np.dot(a1, np.power(x[(0, :)], 2).T)
        f2 = np.dot(a2, -(np.power(x[(0, :)], 3).T))
        obj = np.hstack((f1, f2))
        for i in range(1, len(x)):
            f1 = np.dot(a1, np.power(x[(i, :)], 2).T)
            f2 = np.dot(a2, -(np.power(x[(i, :)], 3).T))
            obj = np.vstack((obj, np.hstack((f1, f2))))
        return np.array(obj)

    
    def individual(self, decs):
        '''
        turn decision vectors into individuals
        :param decs: decision vectors
        :return: individuals
        '''
        pop_obj = self.cost_fun(decs)
        return [decs,pop_obj]

    
    def unit_population(self, population, offspring):
        '''
        Combine two population
        :param population: parent population
        :param offspring: offspring population
        :return:
        '''
        return [
            np.vstack((population[0], offspring[0])),
            np.vstack((population[1], offspring[1]))]

    
    def initialize(self):
        '''
        initialize the population
        :return: the initial population
        '''
        pop_dec = np.random.random((self.N, self.d)) * (self.upper - self.lower) + self.lower
        return self.individual(pop_dec)

    initialize = property(initialize)
    
    def variation(self, pop_dec):
        '''
        Generate offspring individuals
        :param pop_dec: decision vectors
        :return:
        '''
        pro_c = 1
        dis_c = 20
        pro_m = 1
        dis_m = 20
        pop_dec = pop_dec[:(len(pop_dec) // 2) * 2][:]
        (n, d) = np.shape(pop_dec)
        parent_1_dec = pop_dec[(:n // 2, :)]
        parent_2_dec = pop_dec[(n // 2:, :)]
        beta = np.zeros((n // 2, d))
        mu = np.zeros((n // 2, d))
        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (dis_c + 1))
        beta[mu > 0.5] = np.power(2 * mu[mu > 0.5], -1 / (dis_c + 1))
        beta = beta * -1 * np.random.randint(2, (n // 2, d), **None)
        beta[np.random.random((n // 2, d)) < 0.5] = 1
        beta[np.tile(np.random.random((n // 2, 1)) > pro_c, (1, d))] = 1
        offspring_dec = np.vstack(((parent_1_dec + parent_2_dec) / 2 + beta * (parent_1_dec - parent_2_dec) / 2, (parent_1_dec + parent_2_dec) / 2 - beta * (parent_1_dec - parent_2_dec) / 2))
        site = np.random.random((n, d)) < pro_m / d
        mu = np.random.random((n, d))
        temp = site & (mu <= 0.5)
        lower = np.tile(self.lower, (n, 1))
        upper = np.tile(self.upper, (n, 1))
        norm = (offspring_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
        offspring_dec[temp] += (upper[temp] - lower[temp]) * (np.power(2 * mu[temp] + (1 - 2 * mu[temp]) * np.power(1 - norm, dis_m + 1), 1 / (dis_m + 1)) - 1)
        temp = site & (mu > 0.5)
        norm = (upper[temp] - offspring_dec[temp]) / (upper[temp] - lower[temp])
        offspring_dec[temp] += (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * np.power(1 - norm, dis_m + 1), 1 / (dis_m + 1)))
        offspring_dec = np.maximum(np.minimum(offspring_dec, upper), lower)
        return self.individual(offspring_dec)
