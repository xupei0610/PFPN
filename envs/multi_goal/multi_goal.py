import numpy as np
import matplotlib.pyplot as plt

from gym.utils import EzPickle
from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv

class OneDimensionBandit(MujocoEnv, EzPickle):

    def __init__(self):
        self._render = False
        self.rng = np.random.RandomState()
        self.noise = 0
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1, ),
            dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1, ),
            dtype=np.float32)
        self.goals = [-0.25, 0.75]
        EzPickle.__init__(**locals())

    def seed(self, s):
        self.rng.seed(s)

    def reset(self):
        self._loc = np.zeros(self.observation_space.shape)
        self._loc += self.noise*self.rng.randn(*self.observation_space.shape)
        return np.copy(self._loc)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._loc += action + self.noise*self.rng.randn(*self.observation_space.shape)
        self._loc = np.clip(self._loc, self.observation_space.low, self.observation_space.high)
        dist = np.amin(
            [(self._loc - g)**2 for g in self.goals]
        )
        if dist < 0.1:
            done = True
        else:
            done = False
        rew = -dist
        return np.copy(self._loc), rew, done, {}
    
    def render(self, mode='human', *args, **kwargs):
        pass
        #raise NotImplementedError
    

class TwoDimensionBandit(MujocoEnv, EzPickle):
    def __init__(self):
        self._render = False
        self.rng = np.random.RandomState()
        self.noise = 0
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2, ),
            dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2, ),
            dtype=np.float32)
        EzPickle.__init__(**locals())
        self.goals = np.array(
            (
                (0.5, 0.5),
                (-0.5, -0.5),
                (0.5, -0.5),
                (-0.5, 0.5),
            ),
            dtype=np.float32)
        self._ax = None
        self._env_lines = []
        self.xlim = (-1, 1)
        self.ylim = (-1, 1)
    
    def seed(self, s):
        self.rng.seed(s)

    def reset(self):
        self._loc = np.zeros(self.observation_space.shape)
        self._loc += self.noise*self.rng.randn(*self.observation_space.shape)
        self._loc = np.clip(self._loc, self.observation_space.low, self.observation_space.high)
        if self._render:
            self._render_rollouts(self._trajs)
            if self._current_traj:
                self._trajs.append(self._current_traj)
            if len(self._trajs) > 20: self._trajs.pop(0)
            self._current_traj = [np.copy(self._loc)]
        return np.copy(self._loc)
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._loc += action + self.noise*self.rng.randn(*self.observation_space.shape)
        self._loc = np.clip(self._loc, self.observation_space.low, self.observation_space.high)

        dist = np.sqrt(np.amin([
            np.sum((self._loc - g)**2) for g in self.goals
        ]))

        if dist < 0.1:
            done = True
        else:
            done = False
        rew = -dist

        if self._render:
            self._current_traj.append(np.copy(self._loc))
        return np.copy(self._loc), rew, done, {}


    def render(self, mode='human', *args, **kwargs):
        self._render = True
        self._current_traj = []
        self._trajs = []

    def _init_plot(self):
        fig_env = plt.figure(figsize=(10, 10))
        self._ax = fig_env.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = []
        self._ax.set_xlim(self.xlim)
        self._ax.set_ylim(self.ylim)

        self._ax.set_title('Multigoal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')

        self._plot_position_cost(self._ax)

    def _render_rollouts(self, paths=()):
        """Render for rendering the past rollouts of the environment."""
        if self._ax is None:
            self._init_plot()

        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []

        for path in paths:
            positions = np.stack(path)
            xx = positions[:, 0]
            yy = positions[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b')

        plt.draw()
        plt.pause(0.01)

    def _plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        costs = np.sqrt(np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goals
        ], axis=0))

        contours = ax.contour(X, Y, costs, 20)
        ax.clabel(contours, inline=1, fontsize=10)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goals[:, 0],
                       self.goals[:, 1], 'ro')
        return [contours, goal]

