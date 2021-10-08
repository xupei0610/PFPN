# Particle Filtering Policy Network (PFPN)
This code is to support the paper _**PFPN: Continuous Control of Physically Simulated Characters using Particle Filtering Policy Network**_. In the paper, we propose PFPN as a replacement of the policy network with Gaussian policies to solve continuous control problems by adaptively discretizing action spaces. This is a general framework to deep reinforcement learning using policy gradient method and can be easily applied on current widely used on-policy and off-policy policy gradient methods.

PFPN show its advantage in high-dimensional, continuous control tasks, espeically for physics-based character control problems, compared to Guassian policies and the fixed, uniform discretization scheme (DISCRETE).

This paper has been accepted by Motion, Interaction and Games (MIG '21).
[[arXiv Paper Link](https://arxiv.org/abs/2003.06959)]
[[Youtube Video Link](https://www.youtube.com/watch?v=YTtdnq0WpWo)]

Here we provide the implementation of DPPO, A3C, IMPALA and SAC using PFPN. We also provide our implementation of DeepMimic tasks depending on Pybullet library through which the benchmark results shown in the paper can be reproduced.


## Dependencies

    Tensorflow 1.14
    Tensorflow Probability 0.7
    OpenAI Gym
    Pybullet

All of those packages can be installed by

    pip install --user -r requirements.txt


## Usage
To reproduce the benchmark results shown in the paper by the following command:

    bash benchmark.sh ${environment} ${setting_file} ${#_of_particles} ${random_seed} --train

Supported environments include Roboschool environments and our implemented DeepMimic tasks. All setting files are provided in `settings` folder.

For example, to reproduce the benchmark results of `DeepMimicWalk` task with DPPO and 35 particles per action dimension can run

    bash benchmark.sh DeepMimicWalk-v0 deepmimic.deepmimic_dppo_pfpn 35 1 --train

The checkpoint file and log file will be stored at `ckpt_DeepMimicWalk-v0` and `log_DeepMimicWalk-v0` automatically. Use `--debug` option to show optimization information during training.

To visualize the training result can run the above command without `--train` option, i.e.

    bash benchmark.sh DeepMimicWalk-v0 deepmimic.deepmimic_dppo_pfpn 35 1


All benchmark supported environments and the corresponding setting files are listed below.

| Environment                   | Setting File                                    | # of Particles |
|-------------------------------|-------------------------------------------------|----------------|
| DeepMimic{Walk/Punch/Kick}-v0 | deepmimic.deepmimic_{dppo/a3c/impala/sac_async}_pfpn | 35        |    
