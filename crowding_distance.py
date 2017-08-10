import numpy as np


def crowding_distance(pop_obj, front_no):
    N, M = np.shape(pop_obj)
    crowd_dis = np.zeros(N)
    front = np.unique(front_no)
    Fronts = front[front != np.inf]
    for f in range(len(Fronts)):
        Front = np.nonzero(front_no == Fronts[f])[0]
        Fmax = pop_obj[Front, :].max(0)
        Fmin = pop_obj[Front, :].min(0)
        for i in range(M):
            rank = np.argsort(pop_obj[Front, i])
            crowd_dis[Front[rank[0]]] = np.inf
            crowd_dis[Front[rank[-1]]] = np.inf
            for j in range(1,len(Front)-1):
                crowd_dis[Front[rank[j]]] = crowd_dis[Front[rank[j]]]+ (pop_obj[Front[rank[j + 1]], i] - pop_obj[Front[rank[j - 1]], i]) / (Fmax[i] - Fmin[i])
    return crowd_dis

