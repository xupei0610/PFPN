import argparse
import os, sys

import gym
import pybullet_envs

from envs import deepmimic_envs
from envs import multi_goal

import networks
import models

gym.logger.set_level(40)

parser = argparse.ArgumentParser()
parser.add_argument("--setting", type=str, required=True)
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--port", type=int, default=2425)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--device", type=str, default="GPU:0")
parser.add_argument("--save_checkpoint_interval", type=int, default=1000)

parser.add_argument("--particles", type=int, default=None)

settings, _ = parser.parse_known_args()
settings.max_samples = None

external_params = [
    "network_opts", "worker_opts",
    "workers",
    "worker", "network", "model", "max_samples", "save_checkpoint_interval", "max_iterations",
    "env_wrapper"
]
_ = __import__(settings.setting, globals(), locals(), external_params, 0)
if hasattr(_, "argument"):
    _.argument(parser)
    settings = parser.parse_args()
for k in external_params:
    if hasattr(_, k):
        settings.__setattr__(k, getattr(_, k))

settings.log_dir = settings.setting.split(".")[-1]
if settings.suffix:
    settings.suffix = "_" + settings.suffix

if settings.particles is not None:
    settings.network_opts["particles"] = settings.particles
    settings.suffix = "_particle" + str(settings.particles) + settings.suffix

settings.checkpoint_dir = "ckpt_{}/{}{}/{}".format(settings.env, settings.log_dir, settings.suffix, settings.seed)
settings.log_dir = "log_{}/{}{}/{}".format(settings.env, settings.log_dir, settings.suffix, settings.seed)


worker = getattr(models, settings.worker)
network = getattr(networks, settings.network)
model = getattr(models, settings.model)


def env_wrapper(name, render):
    e = gym.make(settings.env)
    if render: e.render()
    if hasattr(settings, "env_wrapper"):
        e = settings.env_wrapper(e)
    return e

def network_wrapper(env, trainable):
    kwargs ={
        "trainable": trainable,
        "state_shape": env.observation_space.shape,
        "action_shape": env.action_space.shape
    }
    if hasattr(env.action_space, "low"):
        kwargs["action_lower_bound"] = env.action_space.low
        kwargs["action_upper_bound"] = env.action_space.high
    net = network(**kwargs, **settings.network_opts)
    return net

def worker_wrapper(**kwargs):
    return worker(**kwargs, **settings.worker_opts)


if model == models.LearnerModel:
    distributions = {
        "learner": [
            "localhost:{}".format(settings.port)
        ],
        "actor": [
            "localhost:{}".format(settings.port + i + 1) for i in range(settings.workers)
        ]
    }
else:
    distributions = {
        "worker": [
            "localhost:{}".format(settings.port + i + 1) for i in range(settings.workers)
        ]
    }
    if settings.workers > 1:
        distributions["ps"] = [
            "localhost:{}".format(settings.port)
        ]
       
model_settings = {
    "save_checkpoint_interval": settings.save_checkpoint_interval,
    "max_iterations": settings.max_iterations,
    "max_samples": settings.max_samples,
    "device": settings.device,
    "seed": settings.seed,
    "log_dir": settings.log_dir,
    "checkpoint_dir": settings.checkpoint_dir,
    "debug": settings.debug
}
 
model = model(
    worker_wrapper, env_wrapper, network_wrapper,
    **model_settings
)

if __name__ == "__main__":
    if settings.train:
        print("\n"*5)
        print("#"*80)
        print("#"*80)
        for k, v in model_settings.items():
            print("{}: {}".format(k, v))
        for k, v in settings.network_opts.items():
            print("{}: {}".format(k, v))
        for k, v in settings.worker_opts.items():
            print("{}: {}".format(k, v))
        print("workers: {}".format(settings.workers))
        print("#"*80)
        print("#"*80)
        print("\n"*5)
        model.start(distributions)
    else:
        model.start()
