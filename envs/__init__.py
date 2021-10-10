from . import deepmimic

__version__ = 0
try:
    import re
    import gym
    re1 = re.compile('(.)([A-Z][a-z]+)')
    re2 = re.compile('([a-z0-9])([A-Z])')
    ver = "-v"+str(__version__)
    print("Loading DeepMimic environments:")
    for c in dir(deepmimic):
        if c.startswith("DeepMimic") and not c.endswith("DeepMimic"):
            gym.envs.registration.register(
                id="DeepMimic"+re2.sub(r"\1_\2", re1.sub(r"\1-\2", c[9:]))+ver,
                entry_point=__name__+".deepmimic:"+c,
                max_episode_steps=600
            )
            print("    DeepMimic"+re2.sub(r"\1_\2", re1.sub(r"\1-\2", c[9:]))+ver)
except ModuleNotFoundError:
    pass
