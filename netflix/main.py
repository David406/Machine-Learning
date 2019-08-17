import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("test_incomplete.txt")

# TODO: Your code here
K = 3
seed = 4
mixture, post = common.init(X, K, seed)
##################### K-means #########################
#mixture, post, cost = kmeans.run(X, mixture, post)
#print(cost)
#################### Naive EM ##########################
#mixture, post, LL = naive_em.run(X, mixture, post)
#print(common.bic(X, mixture, LL))
#print(LL)
###################### EM ##############################
mixture, post, LL = em.run(X, mixture, post)
print(LL)

common.plot(X, mixture, post, "K = %d" % K)
