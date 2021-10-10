import math

class Env(object):

    name = None
    action_space = None
    observation_space = None
    reward_range = (-math.inf, math.inf)

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        return

    def seed(self, seed):
        return
    
    def __str__(self):
        if self.name is None:
            return "<{} instance>".format(type(self).__name__)
        else:
            return self.name
