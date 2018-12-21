from bandit import *
from exp3 import *
import numpy.random as rd


K = 10 # meaning there will be 10 arms to pull
bandits = list()
accum = list()
for i in range(1,K):
    tam_dist = rd.randint(128, size=1)
    base_loc = (rd.rand() + 10.0)*100.0 - 10.0
    base_var = rd.rand() * 5
    bandits.append(bandit(rd.normal(loc=base_loc, scale=base_var, size=tam_dist)))
    accum.append(0)

# Print of series of 10, and their sums:
this_iter = list()
for i in range(1, 10):
    for i,b in enumerate(bandits):
        reward = b.next_reward()
        this_iter.append(reward)
        accum[i] += reward
    #print(this_iter)
    this_iter = list()
#print(accum)

T = 1000 # The simulation will run for 1000 iterations
algo = Exp3(bandits)
my_reward, chosen_sequence, worst_case_regret, optimal_sequence = algo.work(0,T,1)
print("My reward was: ", my_reward)
print("The chosen sequence was:")
print(chosen_sequence)
print("The best sequence was:")
print(optimal_sequence)
print("Overall worst case regret was: ", sum(worst_case_regret))
accuracy = list()
for i,a in enumerate(chosen_sequence):
    accuracy.append(a is optimal_sequence[i])
print("Overall accuracy is: ", sum(accuracy)/T)
