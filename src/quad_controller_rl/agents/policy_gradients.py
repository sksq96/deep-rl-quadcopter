"""Policy search agent."""

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent

import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as slim
import gym
import random
from collections import deque
from numpy.random import normal

class DDPG(BaseAgent):
	"""Sample agent that searches for optimal policy deterministically."""

	def __init__(self, task):
		# Task (environment) information
		self.task = task
		self.state_size = int(np.prod(self.task.observation_space.shape))
		self.action_size = int(np.prod(self.task.action_space.shape))
		# self.state_range = self.task.observation_space.high - self.task.observation_space.low
		# self.action_range = self.task.action_space.high - self.task.action_space.low

		# Score tracker and learning parameters
		# self.best_score = -np.inf

		self.noise_std = 4
		self.noise_rate = 0.995

		self.replay_memory = deque(maxlen=1000000)

		# Network parameters
		self.hidden_size = 16
		self.learn_rate = 1e-3

		self.discount_factor = 0.9
		self.tau = 0.05

		self.state_input_ph = tf.placeholder(tf.float32, shape=(None, self.state_size))
		self.action_input_ph = tf.placeholder(tf.float32, shape=(None, self.action_size))
		self.target_q_ph = tf.placeholder(tf.float32, shape=(None, 1))

		with tf.variable_scope("actor"):
			self.actor_output = self.build_actor(self.state_input_ph)

		with tf.variable_scope("critic"):
			self.critic_output = self.build_critic(self.state_input_ph, self.action_input_ph)

		with tf.variable_scope("target_actor"):
			self.target_actor_output = self.build_actor(self.state_input_ph)

		with tf.variable_scope("target_critic"):
			self.target_critic_output = self.build_critic(self.state_input_ph, self.actor_output)


		actor_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
		critic_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
		target_actor_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
		target_critic_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')

		self.update_target_ops = []
		for i in range(len(actor_weights)):
			update_target_op = target_actor_weights[i].assign(self.tau*actor_weights[i] + (1-self.tau)*target_actor_weights[i])
			self.update_target_ops.append(update_target_op)
		for i in range(len(critic_weights)):
			update_target_op = target_critic_weights[i].assign(self.tau*critic_weights[i] + (1-self.tau)*target_critic_weights[i])
			self.update_target_ops.append(update_target_op)

		self.critic_lose = tf.reduce_mean(tf.square(self.target_q_ph - self.critic_output))
		self.critic_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.critic_lose, var_list=critic_weights)

		self.actor_lose = tf.reduce_mean(-self.target_critic_output)
		self.actor_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.actor_lose, var_list=actor_weights)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


		# Episode variables
		self.reset_episode_vars()

	def reset_episode_vars(self):
		self.last_state = None
		self.last_action = None
		self.total_reward = 0.0
		self.count = 0

	def step(self, state, reward, done):
		# Choose an action
		action = self.act(state) + normal(0, self.noise_std)
		
		# Save experience / reward
		if self.last_state is not None and self.last_action is not None:
			self.total_reward += reward
			self.count += 1

			self.replay_memory.append([self.last_state, self.last_action, reward, state])

		# Learn, if at end of episode
		if done:
			self.learn()
			self.reset_episode_vars()

		self.last_state = state
		self.last_action = action
		return action

	def act(self, state):
		state = np.reshape(np.array(state), (-1, self.state_size))
		action = self.sess.run(self.actor_output, {self.state_input_ph: state})
		return action

		# action = self.task.action_space.sample()
		# return action

	def learn(self):
		self.noise_std *= self.noise_rate
		for _ in range(min(100, len(self.replay_memory) // 256)):
			self.batch_updade(128)
			
		# print("DDPG.learn(): t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(self.count, score, self.best_score, self.noise_scale))


	def build_actor(self, state_input):
		actor_fc_1 = slim.fully_connected(state_input, self.hidden_size, activation_fn=tf.nn.relu)
		actor_fc_2 = slim.fully_connected(actor_fc_1, self.hidden_size, activation_fn=tf.nn.relu)
		actor_fc_3 = slim.fully_connected(actor_fc_2, self.hidden_size, activation_fn=tf.nn.relu)
		actor_fc_4 = slim.fully_connected(actor_fc_3, self.hidden_size, activation_fn=tf.nn.tanh)
		actor_output = slim.fully_connected(actor_fc_4, self.action_size, activation_fn=tf.nn.tanh) * 2
		return actor_output

	def build_critic(self, state_input, action_input):
		critic_input = slim.flatten(tf.concat([state_input, action_input], axis=1))
		critic_fc_1 = slim.fully_connected(critic_input, self.hidden_size, activation_fn=tf.nn.relu)
		critic_fc_2 = slim.fully_connected(critic_fc_1, self.hidden_size, activation_fn=tf.nn.relu)
		critic_fc_3 = slim.fully_connected(critic_fc_2, self.hidden_size, activation_fn=tf.nn.tanh)
		critic_fc_4 = slim.fully_connected(critic_fc_3, self.hidden_size, activation_fn=tf.nn.tanh)
		critic_output = slim.fully_connected(critic_fc_4, 1, activation_fn=None)
		return critic_output


	def sample_from_memory(self, batch_size):
		return random.sample(self.replay_memory, batch_size)

	def batch_updade(self, batch_size):
		batch = self.sample_from_memory(batch_size)
		state_0 = np.reshape(np.vstack([b[0] for b in batch]), (-1, self.state_size))
		action_0 = np.reshape(np.vstack([b[1] for b in batch]), (-1, self.action_size))
		reward_0 = np.reshape(np.vstack([b[2] for b in batch]), (-1, 1))
		state_1 = np.reshape(np.vstack([b[3] for b in batch]), (-1, self.state_size))
		
		action_1 = self.sess.run(self.actor_output, {
				self.state_input_ph:state_1
			}
		)
		q = self.sess.run(self.critic_output, {
				self.state_input_ph: state_1, 
				self.action_input_ph: action_1
			}
		)
		target_q = reward_0 + self.discount_factor*q
		
		lose, _ = self.sess.run([self.critic_lose, self.critic_optimizer], { 
				self.state_input_ph: state_0, 
				self.action_input_ph: action_0,
				self.target_q_ph: target_q
			}
		)

		lose, _ = self.sess.run([self.actor_lose, self.actor_optimizer], {
				self.state_input_ph: state_0
			}
		)
		
		self.sess.run(self.update_target_ops)

