import numpy as np
def nd_sort(pop_obj, n_sort):
    '''
    :param pop_obj: the set
    :param k: number of solutions before the maximum Pareto front
    :return: [FrontNo, MaxFNo]
    '''

    N, M = np.shape(pop_obj)
    a, Loc = np.unique(pop_obj[:, 0], return_inverse=True)
    index = pop_obj[:, 0].argsort()
    pop_obj = pop_obj[index, :]
    front_no = np.inf*np.ones(N)
    max_f_no = 0

    while np.sum(front_no < np.inf) < min(n_sort, N):
        max_f_no = max_f_no + 1
        for i in range(N):
            if front_no[i] == np.inf:
                dominated = False
                for j in range(i-1,0,-1):
                    if front_no[j] == max_f_no:
                        m = 2
                        while (m <= M) and (pop_obj[i][m-1] >= pop_obj[j][m-1]):
                            m = m + 1
                        dominated = m > M
                        if dominated or M == 2:
                            break
                if not dominated:
                    front_no[i] = max_f_no
    front_no = front_no[Loc]
    return front_no, max_f_no


