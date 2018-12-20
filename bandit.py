import itertools as it

class bandit:
    def __init__(self, base_list):
        self.dist = it.cycle([abs((x-min(base_list))/(max(base_list)-min(base_list))) for x in base_list])

    def next_reward(self):
        return next(self.dist)
