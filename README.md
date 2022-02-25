# Particle Filtering Policy Network (PFPN)

This code is to support the paper _**PFPN: Continuous Control of Physically Simulated Characters using Particle Filtering Policy Network**_. [[arXiv](https://arxiv.org/abs/2003.06959)]
[[Youtube](https://www.youtube.com/watch?v=YTtdnq0WpWo)]

This paper has been accepted by *Motion, Interaction and Games* (MIG '21), also *NeurIPS 2021 Deep Reinforcement Learning workshop*.

![](doc/teaser.png)

_**Abstract**_ -- Data-driven methods for physics-based character control using reinforcement learning have been successfully applied to generate high-quality motions. However, existing approaches typically rely on Gaussian distributions to represent the action policy, which can prematurely commit to suboptimal actions when solving high-dimensional continuous control problems for highly-articulated characters. In this paper, to improve the learning performance of physics-based character controllers, we propose a framework that considers a particle-based action policy as a substitute for Gaussian policies. We exploit particle filtering to dynamically explore and discretize the action space, and track the posterior policy represented as a mixture distribution. The resulting policy can replace the unimodal Gaussian policy which has been the staple for character control problems, without changing the underlying model architecture of the reinforcement learning algorithm used to perform policy optimization. We demonstrate the applicability of our approach on various motion capture imitation tasks. Baselines using our particle-based policies achieve better imitation performance and speed of convergence as compared to corresponding implementations using Gaussians, and are more robust to external perturbations during character control.

Here we provide the implementation of DPPO and SAC using PFPN. 
We also provide our implementation of DeepMimic tasks using PyBullet library.


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

For example, to reproduce the benchmark results of `DeepMimicWalk` task with DPPO and 35 particles per action dimension can run

    bash benchmark.sh DeepMimicWalk-v0 deepmimic.deepmimic_dppo_pfpn 35 1 --train

To visualize the training result can run the above command without `--train` option, i.e.

    bash benchmark.sh DeepMimicWalk-v0 deepmimic.deepmimic_dppo_pfpn 35 1


All benchmark supported environments and the corresponding setting files are listed below.

| Environment                   | Setting File                                    | # of Particles |
|-------------------------------|-------------------------------------------------|----------------|
| DeepMimic{Walk/Punch/Kick}-v0 | deepmimic.deepmimic_{dppo/sac}_pfpn  | 35        |    


## Pre-trained Models
We provide three pre-trained models using DPPO in `ckpt_DeepMimicWalk-v0`, `ckpt_DeepMimicPunch-v0` and `ckpt_DeepMimicKick-v0` respectively.

To run the pre-trained models using the following command:

    bash benchmark.sh DeepMimicWalk-v0 deepmimic.deepmimic_dppo_pfpn 35 0
    bash benchmark.sh DeepMimicPunch-v0 deepmimic.deepmimic_dppo_pfpn 35 0
    bash benchmark.sh DeepMimicKick-v0 deepmimic.deepmimic_dppo_pfpn 35 0

## Citation
    @inproceedings{pfpn,
        author = {Xu, Pei and Karamouzas, Ioannis},
        title = {PFPN: Continuous Control of Physically Simulated Characters Using Particle Filtering Policy Network},
        year = {2021},
        isbn = {9781450391313},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3487983.3488301},
        doi = {10.1145/3487983.3488301},
        booktitle = {Motion, Interaction and Games},
        articleno = {7},
        numpages = {12},
        keywords = {character animation, physics-based control, reinforcement learning},
        location = {Virtual Event, Switzerland},
        series = {MIG '21}
    }
