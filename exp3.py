import numpy.random as rd
import math

# Exponential-weight algorithm for Exploration and Exploitation
class Exp3:
    # \param K is the number of bandit arms
    def __init__(self, bandit_arr):
        self.arms = bandit_arr
        self.K = len(bandit_arr)
        self.w = list()
        base_w = [1] * self.K # Weights start as 1
        self.w.append(base_w)

    # \param gama is a real between (0,1]
    # \param t is the current iteration, which indexes the weights
    def probabilities(self, t, gama):
        p = list()
        wsum = sum(self.w[t])
        for wi in self.w[t]:
            prob = (1-gama)*(wi/wsum) + (gama/K)
            p.append(prob)
        return p

    # \param prob_dist probability distribution of indexes, in order
    # \param rand_val value between [0,1) used to determine the index
    # \retval index of the chosen arm
    def choose_index(self, prob_dist, rand_val):
        accum = 0.0
        cur_index = 0
        for i,p in enumerate(prob_dist):
            cur_index = i
            accum += p
            if p > rand_val
                return cur_index
        return cur_index

    # \param Ts is the starting index of iterations
    # \param Te is the ending index of iterations
    # \param gama is a real between (0,1]
    # \retval is the estimated best weak regret
    def work(self, Ts, Te, gama):
        chosen_indexes = list()
        wcr = list() # worst case regrets
        accum_rewards = 0.0
        if len(w) < Ts:
            return -1,[-1],[-1] # Error
        for t in range(Ts,Te):
            p = self.probabilities(t, gama)
            i = self.choose_index(p, rd.rand())
            rewards = list()
            for arm in self.arms:
                rewards.append(arm.next_reward())
            xit = rewards[i]
            rwds = [0] * self.K
            rwds[i] = xit/p[i]
            next_w = list()
            for j,w in enumerate(self.w[t]):
                exponential = (gama * rwds[j])/self.K
                tmp = w * math.exp(exponential)
                next_w.append(tmp)
            w.append(next_w)
            chosen_indexes.append(i)
            wcr.append(max(rewards)-xit)
            accum_rewards += xit
        return accum_rewards, chosen_indexes, wcr
