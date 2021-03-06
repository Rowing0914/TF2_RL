import gym, os
import argparse
from datetime import datetime
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import wrap_deepmind, make_atari, MyWrapper
from tf_rl.common.params import ENV_LIST_NATURE, Parameters
from tf_rl.common.memory import PrioritizedReplayBuffer
from tf_rl.common.utils import AnnealingSchedule, eager_setup, gradient_clip_fn, setup_on_colab
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.train import train_DQN_PER, pretrain_DQfD
from tf_rl.common.networks import Duelling_cartpole as Model
from tf_rl.agents.DQfD import DQfD_cartpole
from tf_rl.agents.DQN import DQN_cartpole

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="CartPole", help="game env type => Atari or CartPole")
# parser.add_argument("--env_name", default="Breakout", help="game title")
parser.add_argument("--seed", default=123, help="seed of randomness")
parser.add_argument("--loss_fn", default="mse", help="types of loss function => MSE or huber_loss")
parser.add_argument("--grad_clip_flg", default="",
                    help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or None")
parser.add_argument("--num_frames", default=10000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=1, type=int, help="a frequency of training occurring in training phase")
parser.add_argument("--eval_interval", default=2500, type=int,
                    help="a frequency of evaluation occurring in training phase")
parser.add_argument("--memory_size", default=5000, type=int,
                    help="memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory")
parser.add_argument("--learning_start", default=100, type=int,
                    help="frame number which specifies when to start updating the agent")
parser.add_argument("--sync_freq", default=1000, type=int, help="frequency of updating a target model")
parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float,
                    help="discount factor => gamma > 1.0 or negative => does not converge!!")
parser.add_argument("--update_hard_or_soft", default="hard",
                    help="types of synchronisation method of target and main models => soft or hard update")
parser.add_argument("--soft_update_tau", default=1e-2, type=float,
                    help="in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!")
parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
parser.add_argument("--epsilon_end", default=0.02, type=float, help="final value of epsilon")
parser.add_argument("--decay_steps", default=3000, type=int, help="a period for annealing a value(epsilon or beta)")
parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
parser.add_argument("--n_step", default=10, help="number for n-step")
parser.add_argument("--L2_reg", default=0.005, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--log_dir", default="../../logs/logs/DQfD/", help="directory for log")
parser.add_argument("--model_dir", default="../../logs/models/DQfD/", help="directory for trained model")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = 195
params.test_episodes = 10
# params.goal = ENV_LIST_NATURE["{}NoFrameskip-v4".format(params.env_name)]
params.alpha = 0.6
params.beta_start = 0.4
params.beta_end = 1.0
params.prioritized_replay_noise = 1e-6

env = MyWrapper(gym.make("CartPole-v0"))

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

now = datetime.now()

if params.google_colab:
    # mount the MyDrive on google drive and create the log directory for saving model and logging using tensorboard
    params.log_dir, params.model_dir, params.log_dir_colab, params.model_dir_colab = setup_on_colab("DQfD",
                                                                                                    params.env_name)
else:
    if params.debug_flg:
        params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DQfD_debug/"
        params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DQfD_debug/"
    else:
        params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DQfD/"
        params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DQfD/"

Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.alpha, n_step=params.n_step)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
anneal_lr = AnnealingSchedule(start=0.0025, end=0.00025, decay_steps=params.decay_steps, decay_type="linear")
optimizer = tf.train.RMSPropOptimizer(anneal_lr.get_value(), 0.99, 0.0, 1e-6)

if params.loss_fn == "huber":
    loss_fn = tf.losses.huber_loss
elif params.loss_fn == "mse":
    loss_fn = tf.losses.mean_squared_error
else:
    assert False, "Choose the loss_fn from either huber or mse"

grad_clip_fn = gradient_clip_fn(flag=params.grad_clip_flg)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
Beta = AnnealingSchedule(start=params.beta_start, end=params.beta_end, decay_steps=params.decay_steps)

if params.debug_flg:
    agent = DQfD_cartpole(Model, optimizer, loss_fn, grad_clip_fn, env.action_space.n, params)
else:
    agent = DQfD_cartpole(Model, optimizer, loss_fn, grad_clip_fn, env.action_space.n, params)

ex_params = Parameters("DQN", params.mode)

if not os.path.isdir(ex_params.model_dir):
    assert False, "You need to have a trained expert in {}".format(ex_params.model_dir)

expert = DQN_cartpole(Model, optimizer, loss_fn, grad_clip_fn, env.action_space.n, ex_params)

# TODO: think about two things below:
# 1. how to deal with the architectural change from pre-train to main-train
# 2. how to deal with the Memory buffer,,, because the paper said we must preset the expert's trajectory so that might need two different buffers.

pretrain_DQfD(expert, agent, env, policy, replay_buffer, reward_buffer, summary_writer, Beta)
train_DQN_PER(agent, env, policy, replay_buffer, reward_buffer, Beta, summary_writer)
