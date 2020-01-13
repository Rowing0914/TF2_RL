
<p align="center"><img src="./assets/logo1.png"></p>

## TF-RL(Reinforcement Learning with Tensorflow: EAGER!!)

[![](https://img.shields.io/badge/TensorFlow-v2.0-blue)](https://github.com/Rowing0914/TF2_RL) [![](https://img.shields.io/badge/Platform-Google%20Colab-lightgrey)](https://github.com/Rowing0914/TF2_RL)

This is the repo for implementing and experimenting the variety of RL algorithms using **Tensorflow Eager Execution**. And, since our Lord Google gracefully allows us to use their precious GPU resources without almost restriction, I have decided to enable most of codes run on **Google Colab**. So, if you don't have GPUs, please feel free to try it out on **Google Colab**

Note: As it is known that **Eager mode** is slower than **Graph Execution** in execution time so that I am using **Eager** for debugging and **Graph** mode for training!! How is it possible?? `@tf.function` allows us to render an annotated function to the **computational graph** at execution.



## Installation

- Install from Pypi(Test)

```shell
# this one
$ pip install --index-url https://test.pypi.org/simple/ --no-deps TF_RL
# or this one
$ pip install -i https://test.pypi.org/simple/ TF-RL
```

- Install from Github source

```shell
git clone https://github.com/Rowing0914/TF_RL.git
cd TF_RL
python setup.py install
```



## Features

- Ready-to-run on Google colab( [Result of DQN](https://github.com/Rowing0914/TF_RL/blob/master/result/DQN/README.md) )

```shell
# you can run on google colab, but make sure that there some restriction on session
# 1. 90 minutes session reflesh
# 2. 12 Hours session reflesh
# Assuming you execute cmds below on Google Colab Jupyter Notebook
$ !git clone https://github.com/Rowing0914/TF_RL.git
$ pip install --index-url https://test.pypi.org/simple/ --no-deps TF_RL
$ %cd TF_RL
$ python3.6 examples/{model_name}/{model_name}_eager_atari.py --mode Atari --env_name={env_name} --google_colab=True

# === Execute On Your Local Machine ===
# My dirty workaroud to avoid breaking the connection to Colab is to execute below on local PC
$ watch -n 3600 python3.6 {your_filename}.py

""" Save this code to {your_filename}.py
import pyautogui
import time

# terminal -> chrome or whatever
pyautogui.hotkey("alt", "tab")
time.sleep(0.5)
# reflesh a page
pyautogui.hotkey("ctrl", "r")
time.sleep(1)
# say "YES" to a confirmation dialogue
pyautogui.hotkey("Enter")
time.sleep(1)
# next page
pyautogui.hotkey("ctrl", "tab")
# check all page reload properly
pyautogui.hotkey("ctrl", "tab")
time.sleep(1)
# switch back to terminal
pyautogui.hotkey("alt", "tab")
time.sleep(0.5)
"""
```



## Implementations

- Please check `tf_rl/examples`, each directory contains its own `README` so please follow it as well!!

- Textbook implementations: R.Sutton's Great Book!

  https://github.com/Rowing0914/TF_RL/tree/master/examples/Sutton_RL_Intro



## Game Envs

### Atari Envs

```python
from tf_rl.common.wrappers import wrap_deepmind, make_atari
from tf_rl.common.params import ENV_LIST_NATURE, ENV_LIST_NIPS


# for env_name in ENV_LIST_NIPS:
for env_name in ENV_LIST_NATURE:
    env = wrap_deepmind(make_atari(env_name))
    state = env.reset()
    for t in range(10):
        # env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        # print(reward, next_state)
        state = next_state
        if done:
            break
    print("{}: Episode finished after {} timesteps".format(env_name, t + 1))
    env.close()
```

### Atari Env with Revertable Wrapper
[[Youtube Demo]](https://www.youtube.com/watch?v=dAo2jn7ElLk&feature=youtu.be)

```python
import time, gym
from tf_rl.common.wrappers import wrap_deepmind, make_atari, ReplayResetEnv

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
env = gym.wrappers.Monitor(env, "./video")
env = ReplayResetEnv(env)

state = env.reset()

for t in range(1, 1000):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    state = next_state

    if t == 300:
        time.sleep(0.5)
        recover_state = env.get_checkpoint_state()

    if (t > 300) and (t % 100 == 0):
        env.recover(recover_state)
        env.step(0)  # 1 extra step to burn the current state on ALE's RAM is required!!
        env.render()
        time.sleep(0.5)

env.close()
```

### CartPole-Pixel(Obs: Raw Pixels in NumpyArray)

```python
import gym
from tf_rl.common.wrappers import CartPole_Pixel

env = CartPole_Pixel(gym.make('CartPole-v0'))
for ep in range(2):
	env.reset()
	for t in range(100):
		o, r, done, _ = env.step(env.action_space.sample())
		print(o.shape)
		if done:
			break
env.close()
```

### MuJoCo(pls, check the MuJoCo official repo for more details...)

```python
# run this from the terminal and make sure you are loading appropriate environment variables
# $ echo $LD_LIBRARY_PATH

import gym
from tf_rl.common.params import DDPG_ENV_LIST

for env_name, goal_score in DDPG_ENV_LIST.items():
	env = gym.make(env_name)
	env.reset()
	for _ in range(100):
		env.render()
		env.step(env.action_space.sample()) # take a random action
```

### MuJoCo Humanoid Maze
https://github.com/Rowing0914/MuJoCo_Humanoid_Maze

```python
import gym
import humanoid_maze # this is the external library(check the link above!!)

env = gym.make('HumanoidMaze-v0')

env.reset()
for _ in range(2000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

```

I have contributed to this project as well.
https://github.com/Breakend/gym-extensions
```python
import gym
from gym_extensions.continuous import mujoco

# available env list: https://github.com/Rowing0914/gym-extensions/blob/mujoco200/tests/all_tests.py
env = gym.make("PusherMovingGoal-v1")

env.reset()
for _ in range(100):
    env.render()
    s, r, d, i = env.step(env.action_space.sample()) # take a random action
    print(s.shape, r, d, i)
env.close()

```


## PC Envs

- OS: Linux Ubuntu LTS 18.04
- Python: 3.x
- GPU: NVIDIA RTX 2080 Max Q Design
- Tensorflow: 2.0.0



### GPU Installation Support

- Check [this link](https://www.tensorflow.org/install/gpu)
- if you encounter some error related to the `unmet dependencies`, pls check [this link](https://devtalk.nvidia.com/default/topic/1043184/cuda-setup-and-installation/cuda-install-unmet-dependencies-cuda-depends-cuda-10-0-gt-10-0-130-but-it-is-not-going-to-be-installed/post/5362151/#5362151)



## References

- [Logomaker](<https://www.logaster.co.uk/?_ga=2.128584591.2087808828.1559775482-1265517291.1559775482>)
- if you get stuck at DQN, you may want to refer to this great guy's entry: <https://adgefficiency.com/dqn-debugging/>
