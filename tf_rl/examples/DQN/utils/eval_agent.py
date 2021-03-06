import numpy as np
import datetime
from tf_rl.common.colab_utils import transfer_log_dirs


def eval_Agent(agent, env, n_trial=1, log_dir=None, google_colab=False):
    """ Evaluate the trained agent """
    all_rewards = list()
    for ep in range(n_trial):
        if ep == 0: env.record_start()
        state = np.array(env.reset())
        done = False
        episode_reward = 0
        while not done:
            # epsilon-greedy for evaluation using a fixed epsilon of 0.05(Nature does this!)
            action = agent.select_action_eval(state, epsilon=0.05)
            next_state, reward, done, _ = env.step(action)
            state = np.array(next_state)
            episode_reward += reward

        if ep == 0: env.record_end()

        all_rewards.append(episode_reward)
        _time = datetime.datetime.now()
        print("[{}] | Evaluation | Ep: {}/{} | Score: {} |".format(_time, ep + 1, n_trial, episode_reward))

    if google_colab: transfer_log_dirs(log_dir)
    return np.array([all_rewards]).mean()
