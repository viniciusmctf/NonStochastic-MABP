from bandit import *
import numpy.random as rd


K = 10 # meaning there will be 10 arms to pull
bandits = list()
accum = list()
for i in range(1,K):
    tam_dist = rd.randint(128, size=1)
    bandits.append(bandit(rd.normal(loc=8.0, scale=1.0, size=tam_dist)))
    accum.append(0)

# Print of series of 10, and their sums:
this_iter = list()
for i in range(1, 10):
    for i,b in enumerate(bandits):
        reward = b.next_reward()
        this_iter.append(reward)
        accum[i] += reward
    print(this_iter)
    this_iter = list()
print(accum)
