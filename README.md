# laser-hockey-env

This repository contains the approaches and report of the Codeyotes for the 2023 laser-hockey reinforcement learning challenge at the University of Tübingen. Our team consisted of Balint, Joschi and Enes. In the final tournament our two competing agents placed in 2nd and 3rd place among all 89 agents participating. They were the two strongest model-free agents, getting only beaten by an implementation based on MuZero. Our report can be found in ``/report``.

## Install

``python3 -m pip install git+https://github.com/martius-lab/laser-hockey-env.git``

or add the following line to your Pipfile

``laserhockey = {editable = true, git = "https://git@github.com/martius-lab/laser-hockey-env.git"}``

or with Conda:

``conda create --name gym-rl``

``conda activate gym-rl``

``conda install pip -c anaconda``

``pip install gymnasium numpy box2d-py jupyter matplotlib torch pygame wandb``

## HockeyEnv

![Screenshot](assets/hockeyenv1.png)

``laserhockey.hockey_env.HockeyEnv``

A two-player (one per team) hockey environment.
For our Reinforcement Learning Lecture @ Uni-Tuebingen.
See Hockey-Env.ipynb notebook on how to run the environment.

The environment can be generated directly as an object or via the gym registry:

``env = gym.envs.make("Hockey-v0")``

There is also a version against the basic opponent (with options)

``env = gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)``



## LaserHockeyEnv

A laser-hockey game implementation within openAI gym using the Box2D engine. It is quite a challenging environment
See Laser-Hockey-Env.ipynb notebook on how to run the environment.
