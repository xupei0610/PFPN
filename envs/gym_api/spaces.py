# a very simple api to keep compatible with basic usage of openai gym

import numpy as np

class Space(object):

    def __init__(self, shape=None):
        self.np_random = np.random.RandomState()
        self.shape = shape

    def contains(self, x):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __eq__(self, rhs):
        raise NotImplementedError


# class Discrete(Space):

#     def __init__(self, n=0):
#         self._n = n
#         super().__init__([n])

#     @property
#     def n(self):
#         return self._n

#     @n.setter
#     def n(self, n):
#         self._n = n
#         self.shape = [n]

#     def contains(self, x):
#         return -1 < int(x) < self._n

#     def sample(self):
#         return self.np_random.randint(self._n)

#     def __repr__(self):
#         return "Discrete({})".format(self._n)

#     def __eq__(self, rhs):
#         return self._n == rhs.n


class Box(Space):
    
    def __init__(self, low=None, high=None, shape=None):
        self.low = low
        self.high = high
        if shape is None:
            shape = np.array(self.low).shape
        super().__init__(shape)
    
    def sample(self):
        return self.np_random.uniform(self.low, self.high)

    def __repr__(self):
        return "Box({})".format(self.shape)
