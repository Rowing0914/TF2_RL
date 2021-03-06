import argparse
import itertools
import time
from collections import deque

import numpy as np
import tensorflow as tf

from tf_rl.experiments.Exploration_Strategies.utils import eval_Agent, make_grid_env
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess
from tf_rl.common.utils import eager_setup, get_ready, soft_target_model_update_eager, logger

eager_setup()
KERNEL_INIT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
L2 = tf.keras.regularizers.l2(1e-2)


# define the actor and critic
class Actor(tf.keras.Model):
    def __init__(self, num_action=1):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(20, activation='relu', kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(10, activation='relu', kernel_initializer=KERNEL_INIT)
        self.pred = tf.keras.layers.Dense(num_action, activation='tanh', kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        pred = self.pred(x)
        return pred


class Critic(tf.keras.Model):
    def __init__(self, output_shape):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.pred = tf.keras.layers.Dense(output_shape, activation='linear', kernel_regularizer=L2, bias_regularizer=L2,
                                          kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, obs, act):
        x = self.dense1(obs)
        x = self.dense2(tf.concat([x, act], axis=-1))
        pred = self.pred(x)
        return pred


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Cont-GridWorld-v2", type=str, help="Env title")
parser.add_argument("--train_flg", default="original", type=str, help="train flg: original or on-policy")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
# parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--num_frames", default=500_000, type=int, help="total frame in a training")
parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
parser.add_argument("--batch_size", default=200, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update tau")
parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--mu", default=0.3, type=float, help="magnitude of randomness")
parser.add_argument("--sigma", default=0.2, type=float, help="magnitude of randomness")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 1

params.log_dir = "./logs/logs/DDPG-seed{}/".format(params.seed)
params.actor_model_dir = "./logs/models/DDPG-seed{}/actor/".format(params.seed)
params.critic_model_dir = "./logs/models/DDPG-seed{}/critic/".format(params.seed)
params.video_dir = "./logs/video/DDPG-seed{}/".format(params.seed)
params.plot_path = "./logs/plots/DDPG-seed{}/".format(params.seed)

env = make_grid_env(plot_path=params.plot_path)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=0.15, mu=params.mu, sigma=params.sigma)
agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)

get_ready(agent.params)

global_timestep = tf.compat.v1.train.get_or_create_global_step()
time_buffer = deque(maxlen=agent.params.reward_buffer_ep)
log = logger(agent.params)

traj = list()

with summary_writer.as_default():
    # for summary purpose, we put all codes in this context
    with tf.contrib.summary.always_record_summaries():

        for i in itertools.count():
            state = env.reset()
            total_reward = 0
            self_rewards = 0
            start = time.time()
            agent.random_process.reset_states()
            done = False
            episode_len = 0
            while not done:
                traj.append(state)
                if global_timestep.numpy() < agent.params.learning_start:
                    action = env.action_space.sample()
                else:
                    action = agent.predict(state)
                next_state, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                replay_buffer.add(state, action, reward, next_state, done)

                """
                === Update the models
                """
                if global_timestep.numpy() > agent.params.learning_start:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
                    loss = agent.update(states, actions, rewards, next_states, dones)
                    soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.soft_update_tau)
                    soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.soft_update_tau)

                global_timestep.assign_add(1)
                episode_len += 1
                total_reward += reward
                state = next_state

                # for evaluation purpose
                if global_timestep.numpy() % agent.params.eval_interval == 0:
                    agent.eval_flg = True

            """
            ===== After 1 Episode is Done =====
            """
            # save the updated models
            agent.actor_manager.save()
            agent.critic_manager.save()

            # store the episode related variables
            reward_buffer.append(total_reward)
            time_buffer.append(time.time() - start)

            # logging on Tensorboard
            tf.contrib.summary.scalar("reward", total_reward, step=global_timestep.numpy())
            tf.contrib.summary.scalar("exec time", time.time() - start, step=global_timestep.numpy())
            if i >= agent.params.reward_buffer_ep:
                tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=global_timestep.numpy())

            # logging
            if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss), 0, [0])

            # evaluation
            if agent.eval_flg:
                env.vis_exploration(traj=np.array(traj),
                                    file_name="exploration_train_DDPG_{}.png".format(global_timestep.numpy()))
                env.vis_trajectory(traj=np.array(traj), file_name="traj_train_DDPG_{}.png".format(global_timestep.numpy()))
                eval_Agent(env, agent)
                agent.eval_flg = False

            # check the stopping condition
            if global_timestep.numpy() > agent.params.num_frames:
                print("=== Training is Done ===")
                traj = np.array(traj)
                env.vis_exploration(traj=traj, file_name="exploration_DDPG_during_training.png")
                env.vis_trajectory(traj=traj, file_name="traj_DDPG_during_training.png")
                eval_Agent(env, agent)
                env.close()
                break
