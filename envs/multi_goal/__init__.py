
import gym

gym.envs.registration.register(
    id="OneDimensionBandit-v0",
    entry_point=__name__+".multi_goal:OneDimensionBandit",
    max_episode_steps=10
)
gym.envs.registration.register(
    id="TwoDimensionBandit-v0",
    entry_point=__name__+".multi_goal:TwoDimensionBandit",
    max_episode_steps=10
)
