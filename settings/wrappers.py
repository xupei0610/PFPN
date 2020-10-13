class DiscreteActionWrapper:
    def __init__(self, env, n):
        self.env = env
        self.action_cont = [
            # [l + (_+0.5)*(h-l)/n for _ in range(n)]
            # [l + _*(h-l)/(n+1) for _ in range(n)]
            [l + _*(h-l)/(n-1) for _ in range(n)]
            for h, l in zip(self.env.action_space.high, self.env.action_space.low)
        ]
        self.env.action_space.shape = [n] * len(self.env.action_space.high)
        delattr(self.env.action_space, "low")
        delattr(self.env.action_space, "high")
    def step(self, a):
        action_cont = [_[i] for _, i in zip(self.action_cont, a)]
        return self.env.step(action_cont)
    def __getattr__(self, name):
        return getattr(self.env, name)
