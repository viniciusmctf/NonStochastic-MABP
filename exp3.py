import numpy.random as rd
import math

# Exponential-weight algorithm for Exploration and Exploitation
class Exp3:
    # \param bandit_arr is the array of arms to pull
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
            prob = (1-gama)*(wi/wsum) + (gama/float(self.K))
            p.append(prob)
        return p

    # \param prob_dist probability distribution of indexes, in order
    # \param rand_val value between [0,1) used to determine the index
    # \retval index of the chosen arm
    def choose_index(self, prob_dist, rand_val):
        accum = 0.0
        cur_index = 0
        for i,p in enumerate(prob_dist):
            # print(rand_val, accum, p)
            cur_index = i
            accum += p
            if accum > rand_val:
                return cur_index
        return cur_index

    # \param Ts is the starting index of iterations
    # \param Te is the ending index of iterations
    # \param gama is a real between (0,1]
    # \retval is the estimated best weak regret
    def work(self, Ts, Te, gama):
        chosen_indexes = list()
        optimal_sequence = list()
        wcr = list() # worst case regrets
        accum_rewards = 0.0
        if len(self.w) < Ts:
            return -1,[-1],[-1] # Error
        for t in range(Ts,Te):
            p = self.probabilities(t, gama)
            i = self.choose_index(p, rd.rand())
            rewards = list()
            best_case = 0.0
            bci = 0
            for index,arm in enumerate(self.arms):
                tmp = arm.next_reward()
                if best_case < tmp:
                    best_case = tmp
                    bci = index
                rewards.append(tmp)
            xit = rewards[i]
            rwds = [0] * self.K
            rwds[i] = xit/p[i]
            next_w = list()
            for j,wj in enumerate(self.w[t]):
                exponential = (gama * rwds[j])/self.K
                tmp = wj * math.exp(exponential)
                next_w.append(tmp)
            self.w.append(next_w)
            optimal_sequence.append(bci)
            chosen_indexes.append(i)
            wcr.append(max(rewards)-xit)
            accum_rewards += xit
        return accum_rewards, chosen_indexes, wcr, optimal_sequence
