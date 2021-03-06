from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class CEMOptimizer(Optimizer):

    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.cost_function = cost_function

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var):
        """[Tensorflow Eager compatible]
        Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = tf.cast(init_mean, dtype=tf.float32), tf.cast(init_var, dtype=tf.float32), 0
        tf_dist = tfd.TruncatedNormal(low=-2, high=2, loc=tf.zeros_like(mean), scale=tf.ones_like(var))

        def _cond(mean, var, t):
            bool1 = tf.less(t, self.max_iters)
            bool2 = tf.less(self.epsilon, tf.reduce_max(var))
            return tf.math.logical_and(x=bool1, y=bool2)

        def _body(mean, var, t):
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = tf.math.minimum(
                tf.math.minimum(tf.math.square(lb_dist / 2), tf.math.square(ub_dist / 2)),
                var)
            constrained_var = tf.cast(constrained_var, dtype=tf.float32)

            samples = tf_dist.sample(sample_shape=self.popsize)
            samples = tf.cast(samples, tf.float32) * tf.math.sqrt(constrained_var) + mean

            costs = self.cost_function(samples)

            elites = tf.gather(samples, tf.argsort(costs))[:self.num_elites]

            new_mean = tf.math.reduce_mean(elites, axis=0)
            new_var = tf.math.reduce_variance(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            t += 1
            return mean, var, t

        mean, var, t = tf.while_loop(cond=_cond, body=_body, loop_vars=[mean, var, t])

        return mean