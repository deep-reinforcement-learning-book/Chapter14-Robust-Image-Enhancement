import argparse
import os
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from env import Env


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


class Agent(object):
    """PPO Agent"""
    def __init__(self, feature, actor, critic, optimizer,
                 epsilon=0.1, gamma=0.95, c1=1.0, c2=1e-4, gae_lambda=0.95):
        """
        Args:
            feature (tf.keras.Model): backbone of actor and critic
            actor (tf.keras.Model): actor network
            critic (tf.keras.Model): critic network
            optimizer (tf.keras.optimizers.Optimizer): optimizer for NNs
            epsilon (float): epsilon in clip
            gamma (float): reward discount
            c1 (float): factor of value loss
            c2 (float): factor of entropy
        """
        self.feature, self.actor, self.critic = feature, actor, critic
        self.optimizer = optimizer

        self._epsilon = epsilon
        self.gamma = gamma
        self._c1 = c1
        self._c2 = c2
        self.gae_lambda = gae_lambda

    def act(self, state, greedy=False):
        """
        Args:
            state (numpy.array): 1 * 4048
            greedy (bool): whether select action greedily

        Returns:
            action (int): selected action
            logprob (float): log prob of the selected action
            value (float): value of the current state
        """
        feature = self.feature(state)
        logprob = self.actor(feature)
        if greedy:
            action = tf.argmax(logprob[0]).numpy()
            return action, 0, 0
        else:
            value = self.critic(feature)
            logprob = logprob[0].numpy()
            action = np.random.choice(range(len(logprob)), p=np.exp(logprob))
            return action, logprob[action], value.numpy()[0, 0]

    def sample(self, env, sample_episodes, greedy=False):
        """ Sample trajectories from given env
        Args:
            env: environment
            sample_episodes (int): how many episodes will be sampled
            greedy (bool): whether select action greedily
        """
        trajectories = []  # s, a, r, logp
        e_reward = 0
        e_reward_max = 0
        for _ in range(sample_episodes):
            s = env.reset()
            values = []
            while True:
                a, logp, v = self.act(s, greedy)
                s_, r, done, info = env.step(a)
                e_reward += r
                values.append(v)
                trajectories.append([s, a, r, logp, v])
                s = s_
                if done:
                    e_reward_max += info['max_reward']
                    break
            episode_len = len(values)
            gae = np.empty(episode_len)
            reward = trajectories[-1][2]
            gae[-1] = last_gae = reward - values[-1]
            for i in range(1, episode_len):
                reward = trajectories[-i - 1][2]
                delta = reward + self.gamma * values[-i] - values[-i - 1]
                gae[-i - 1] = last_gae = \
                    delta + self.gamma * self.gae_lambda * last_gae
            for i in range(episode_len):
                trajectories[-(episode_len - i)][2] = gae[i] + values[i]
        e_reward /= sample_episodes
        e_reward_max /= sample_episodes
        return trajectories, e_reward, e_reward_max

    def _train_func(self, b_s, b_a, b_r, b_logp_old, b_v_old):
        all_params = self.feature.trainable_weights + \
                     self.actor.trainable_weights + \
                     self.critic.trainable_weights
        with tf.GradientTape() as tape:
            b_feature = self.feature(b_s)
            b_logp, b_v = self.actor(b_feature), self.critic(b_feature)

            entropy = -tf.reduce_mean(
                tf.reduce_sum(b_logp * tf.exp(b_logp), axis=-1))
            b_logp = tf.gather(b_logp, b_a, axis=-1, batch_dims=1)
            adv = b_r - b_v_old
            adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

            c_b_v = b_v_old + tf.clip_by_value(b_v - b_v_old,
                                               -self._epsilon, self._epsilon)
            vloss = 0.5 * tf.reduce_max(tf.stack(
                [tf.pow(b_v - b_r, 2), tf.pow(c_b_v - b_r, 2)], axis=1), axis=1)
            vloss = tf.reduce_mean(vloss)

            ratio = tf.exp(b_logp - b_logp_old)
            clipped_ratio = tf.clip_by_value(
                ratio, 1 - self._epsilon, 1 + self._epsilon)
            pgloss = -tf.reduce_mean(tf.reduce_min(tf.stack(
                [clipped_ratio * adv, ratio * adv], axis=1), axis=1))

            total_loss = pgloss + self._c1 * vloss - self._c2 * entropy
        grad = tape.gradient(total_loss, all_params)
        self.optimizer.apply_gradients(zip(grad, all_params))
        return entropy

    def optimize(self, trajectories, opt_iter):
        """ Optimize based on given trajectories """
        b_s, b_a, b_r, b_logp_old, b_v_old = zip(*trajectories)
        b_s = np.concatenate(b_s, 0)
        b_a = np.expand_dims(np.array(b_a, np.int64), 1)
        b_r = np.expand_dims(np.array(b_r, np.float32), 1)
        b_logp_old = np.expand_dims(np.array(b_logp_old, np.float32), 1)
        b_v_old = np.expand_dims(np.array(b_v_old, np.float32), 1)
        b_s, b_a, b_r, b_logp_old, b_v_old = map(
            tf.convert_to_tensor, [b_s, b_a, b_r, b_logp_old, b_v_old])
        for _ in range(opt_iter):
            entropy = self._train_func(b_s, b_a, b_r, b_logp_old, b_v_old)

        return entropy.numpy()

    def save(self, path):
        """
        save trained weights
        :return: None
        """
        self.feature.save_weights(os.path.join(path, 'ppo_feature.ckpt'))
        self.actor.save_weights(os.path.join(path, 'ppo_actor.ckpt'))
        self.critic.save_weights(os.path.join(path, 'ppo_critic.ckpt'))

    def load(self, path):
        """
        load trained weights
        :return: None
        """
        self.feature.load_weights(os.path.join(path, 'ppo_feature.ckpt'))
        self.actor.load_weights(os.path.join(path, 'ppo_actor.ckpt'))
        self.critic.load_weights(os.path.join(path, 'ppo_critic.ckpt'))


def get_networks():
    dense = tf.keras.layers.Dense

    # feature
    feature = tf.keras.Sequential([
        dense(2048, activation=tf.nn.relu, input_shape=(2048 + 2000, )),
        dense(512, activation=tf.nn.relu),
        dense(128, activation=tf.nn.relu)
    ])

    # critic
    critic = tf.keras.Sequential([
        dense(1, activation=None, input_shape=(128, ))
    ])

    # actor
    actor = tf.keras.Sequential([
        dense(13, activation=tf.nn.log_softmax, input_shape=(128, ))
    ])

    return feature, actor, critic


def train(option):
    """ Train code """
    print(option)
    if not os.path.exists(option.checkpoint_dir):
        os.mkdir(option.checkpoint_dir)

    train_pairs = []
    with open(option.train_path) as f:
        for line in f:
            train_pairs.append(line.strip().split('\t'))
    train_env = Env(train_pairs)

    feature, actor, critic = get_networks()
    optimizer = tf.optimizers.Adam(option.lr,
                                   clipnorm=option.clipnorm, epsilon=1e-5)
    agent = Agent(feature, actor, critic, optimizer,
                  gae_lambda=option.gae_lambda, c2=option.c2)

    for i_iter in range(1, option.max_iter + 1):
        trajectories, r, maxr = agent.sample(train_env, option.episode_per_iter)
        length = len(trajectories) * 1.0 / option.episode_per_iter
        entropy = agent.optimize(trajectories, option.opt_per_iter)
        print('{} Iter {} Reward {:.4f}/{:.4f} entropy {:.4f} |T| {:.2f}'
              .format(datetime.now(), i_iter, r, maxr, entropy, length))

        if i_iter % 50 == 0:
            agent.save(option.checkpoint_dir)


def valid(option):
    """ Valid """
    pairs = []
    with open(option.valid_path) as f:
        for line in f:
            pairs.append(line.strip().split('\t'))
    n = len(pairs)
    feature, actor, critic = get_networks()
    agent = Agent(feature, actor, critic, None)
    agent.load(option.checkpoint_path)
    reward = 0
    step = 0
    max_reward = 0
    scores = []
    for pair in tqdm(pairs):
        env = Env([pair])
        trajectories, e_r, e_r_max = agent.sample(env, 1, True)
        step += len(trajectories)
        reward += e_r
        max_reward += e_r_max
        if step > 5:
            scores.append((reward, pair, env._rgb_state))
    reward, step, max_reward = reward / n, step / n, max_reward / n
    print('Reward {:.4f} step {:.4f} max_reward {:.4f}'
          .format(reward, step, max_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image enhancement arguments')
    parser.add_argument('--mode', default='train', help='train | valid')
    parser.add_argument(
        '--train_path', type=str, default='train_pairs',
        help='each line is a pair of raw and retouched path splited by \\t')
    parser.add_argument(
        '--valid_path', type=str, default='valid_pairs',
        help='each line is a pair of raw and retouched path splited by \\t')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--episode_per_iter', type=int, default=4,
                        help='how many episode will sample per iter')
    parser.add_argument('--opt_per_iter',
                        help='optimize per sample', type=int, default=2)
    parser.add_argument('--max_iter',
                        help='training iter number', type=int, default=10000)
    parser.add_argument('--lr', type=float,
                        default=1e-5, help='initial learning rate')
    parser.add_argument('--clipnorm',
                        help='clip norm', type=float, default=1.0)
    parser.add_argument('--gae_lambda',
                        help='gae lambda', type=float, default=0.95)
    parser.add_argument('--c2', help='entropy factor', type=float, default=0.01)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
    opt = parser.parse_args()
    if opt.mode == 'train':  # expected maximum reward: 1.7
        train(opt)
    elif opt.mode == 'valid':
        valid(opt)
