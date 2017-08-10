import numpy as np
def tournament(K,N,**kwargs):
    '''
    index = kwargs[:, 0].argsort()
    index = np.argsort(index)
    n = len(kwargs)
    parents = np.random.randint(n, size= (K,N))
    y = index[parents]
    indice = []
    for i in range(n):
        x = np.argmin(y[:,i])
        np.append(indice,x)
    return parents[indice + np.dot(list(range(N)),K)]
    '''
    n = len(kwargs)
    mate = []
    for i in range(N):
        a = np.random.randint(n)
        for j in range(K):
            b = np.random.randint(n)
            for r in range(kwargs[0,:].size):
                if kwargs[b,r] < kwargs[a,r]:
                    a = b.copy()
                    break
                elif kwargs[b,r] == kwargs[a,r]:
                    pass
                else:
                    break
        np.append(mate,a)
    return mate
