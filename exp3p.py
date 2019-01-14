import numpy.random as rd
import math

class Exp3P:
    # \param bandit_arr is the array of arms to pull
    # \param alpha is a real > 0
    # \param gama is a real in (0,1]
    # \param T is the expected time horizon
    def __init__(self, bandit_arr, alpha, gama, T):
        self.arms = bandit_arr
        self.K = len(bandit_arr)
        self.alpha = float(alpha)
        self.gama = float(gama)
        self.T = T
        if self.alpha <= 0:
            self.alpha = 2*math.sqrt(self.K*self.T)
        if self.gama <= 0 or self.gama > 1:
            self.gama = rd.rand()%1 # Force a value inside the interval
        base_w = math.exp((alpha*gama/3.0)*math.sqrt(self.T/self.K))
        self.w = list()
        self.w.append([base_w] * self.K)

    # \param t is the current iteration, which indexes the weights
    def probabilities(self, t):
        p = list()
        wsum = sum(self.w[t])
        for wi in self.w[t]:
            prob = (1-self.gama)*(wi/wsum) + (self.gama/float(self.K))
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
            if accum > rand_val:
                return cur_index
        return cur_index

    # This time, T has to be predefined in order for things to work out
    def work(self):
        chosen_indexes = list()
        optimal_sequence = list()
        wcr = list()
        accum_rewards = 0.0
        for t in range(0,self.T):
            # print("Calculate Probabilities for iteration " + str(t))
            p = self.probabilities(t)
            i = self.choose_index(p, rd.rand())
            rewards = list()
            best_case = 0.0
            bci = 0 # best case index
            for index,arm in enumerate(self.arms):
                tmp = arm.next_reward()
                if best_case < tmp:
                    best_case = tmp
                    bci = index
                rewards.append(tmp)
            xit = rewards[i] # xit is the reward of the chosen index
            rwds = [0] * self.K
            rwds[i] = xit/p[i]
            next_w = list()
            active_w = list()
            # Guarantee w[t] is safe to work with. If not, normalize weights.
            # Evade Python Overflow
            if sum(self.w[t]) > 1e+250:
                for j,wj in enumerate(self.w[t]):
                    active_w.append(wj%self.w[0][j])
            else:
                active_w = self.w[t]

            for j,wj in enumerate(active_w):
                tmp = self.alpha/(p[j] * math.sqrt(self.K * self.T))
                expo = ((self.gama/3*self.K) * (rwds[j] + (tmp)))
                # print(expo, wj)
                next_w.append(wj*math.exp(expo))
            self.w.append(next_w)
            # print(next_w)
            optimal_sequence.append(bci)
            chosen_indexes.append(i)
            wcr.append(max(rewards) - xit)
            accum_rewards += xit
        return accum_rewards, chosen_indexes, wcr, optimal_sequence
