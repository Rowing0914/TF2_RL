import gym
import argparse
import os
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper, wrap_deepmind, make_atari
from tf_rl.common.params import Parameters
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.train import train_DRQN
from tf_rl.agents.DQN import DQN

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
tf.random.set_random_seed(123)


class Model(tf.keras.Model):
    def __init__(self, env_type, num_action):
        super(Model, self).__init__()
        self.env_type = env_type
        if self.env_type == "CartPole":
            self.dense1 = tf.keras.layers.Dense(16, activation='relu')
            self.dense2 = tf.keras.layers.Dense(16, activation='relu')
            self.dense3 = tf.keras.layers.Dense(16, activation='relu')
            self.lstm = tf.keras.layers.LSTM(400, activation='linear', return_sequences=True)
            self.pred = tf.keras.layers.Dense(num_action, activation='linear')
        elif self.env_type == "Atari":
            self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
            self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
            self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
            self.flat = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(512, activation='relu')
            self.pred = tf.keras.layers.Dense(num_action, activation='linear')

    def call(self, inputs):
        if self.env_type == "CartPole":
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = self.dense3(x)
            # print(x.shape)
            # x = self.lstm(tf.reshape(x, [1, 1, 16]))
            x = self.lstm(x)
            pred = self.pred(x)
            return pred
        elif self.env_type == "Atari":
            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flat(x)
            x = self.fc1(x)
            pred = self.pred(x)
            return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="Atari", help="game env type")
    parser.add_argument("--loss_fn", default="MSE", help="types of loss function => MSE or huber_loss")
    parser.add_argument("--grad_clip_flg", default="by_value",
                        help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or None")
    parser.add_argument("--num_frames", default=10_000_000, type=int, help="total frame in a training")
    parser.add_argument("--train_interval", default=4, type=int,
                        help="a frequency of training occurring in training phase")
    parser.add_argument("--memory_size", default=400_000, type=int,
                        help="memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory")
    parser.add_argument("--learning_start", default=10_000, type=int,
                        help="frame number which specifies when to start updating the agent")
    parser.add_argument("--sync_freq", default=5_000, type=int, help="frequency of updating a target model")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size of each iteration of update")
    parser.add_argument("--gamma", default=0.99, type=int,
                        help="discount factor => gamma > 1.0 or negative => does not converge!!")
    parser.add_argument("--update_hard_or_soft", default="hard",
                        help="types of synchronisation method of target and main models => soft or hard update")
    parser.add_argument("--soft_update_tau", default=1e-2, type=float,
                        help="in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!")
    parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
    parser.add_argument("--epsilon_end", default=0.02, type=float, help="final value of epsilon")
    parser.add_argument("--decay_steps", default=100_000, type=int,
                        help="a period for annealing a value(epsilon or beta)")
    parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
    parser.add_argument("--log_dir", default="../../logs/logs/DQN/", help="directory for log")
    parser.add_argument("--model_dir", default="../../logs/models/DQN/", help="directory for trained model")
    args = parser.parse_args()

    try:
        os.system("rm -rf {}".format(args.log_dir))
    except:
        pass

    # I know this is not beautiful, but for the sake of ease of dev and finding the best params,
    # i will leave this for a while
    # TODO: you need to amend this design to the one only args, instead of params
    params = Parameters(algo="DQN", mode=args.mode)
    if args.mode == "Atari":
        params.loss_fn = args.loss_fn
        params.grad_clip_flg = args.grad_clip_flg
        params.num_frames = args.num_frames
        params.memory_size = args.memory_size
        params.learning_start = args.learning_start
        params.sync_freq = args.sync_freq
        params.batch_size = args.batch_size
        params.gamma = args.gamma
        params.update_hard_or_soft = args.update_hard_or_soft
        params.soft_update_tau = args.soft_update_tau
        params.epsilon_start = args.epsilon_start
        params.epsilon_end = args.epsilon_end
        params.decay_steps = args.decay_steps
        params.decay_type = args.decay_type

    if args.mode == "CartPole":
        env = MyWrapper(gym.make("CartPole-v0"))
    elif args.mode == "Atari":
        env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))

    replay_buffer = ReplayBuffer(params.memory_size)
    agent = DQN(args.mode, Model, Model, env.action_space.n, params, args.model_dir)
    Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
                                decay_steps=params.decay_steps)
    policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)

    reward_buffer = deque(maxlen=params.reward_buffer_ep)
    summary_writer = tf.contrib.summary.create_file_writer(args.log_dir)
    train_DRQN(agent, env, policy, replay_buffer, reward_buffer, params, summary_writer)
