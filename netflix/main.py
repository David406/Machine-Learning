import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K = 3
seed = 4
mixture, post = common.init(X, K, seed)
#mixture, post, cost = kmeans.run(X, mixture, post)
#print(cost)
mixture, post, LL = naive_em.run(X, mixture, post)
print(common.bic(X, mixture, LL))
print(LL)
common.plot(X, mixture, post, "K = %d" % K)
