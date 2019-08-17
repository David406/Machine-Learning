import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

# TODO: Your code here
K = 12
seed = 1
mixture, post = common.init(X, K, seed)
##################### K-means #########################
#mixture, post, cost = kmeans.run(X, mixture, post)
#print(cost)
#################### Naive EM #########################
#mixture, post, LL = naive_em.run(X, mixture, post)
#print(common.bic(X, mixture, LL))
#print(LL)
############### EM Matrix Completion ##################
mixture, post, LL = em.run(X, mixture, post)
X_pred = em.fill_matrix(X, mixture)
RMSE = common.rmse(X_gold, X_pred)
#print(LL)
print(RMSE)


#common.plot(X, mixture, post, "K = %d" % K)
