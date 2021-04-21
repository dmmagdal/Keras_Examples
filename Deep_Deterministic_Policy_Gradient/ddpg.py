# ddpg.py
# Implement DDPG algorithm on the Inverted Pendulum problem.
# Deep Deterministic Policy Gradient (DDPG) is a model-free off-policy
# algorithm for learning continuous actions. It combines ideas from DPG
# (Deterministic Policy Gradient) and DQN (Deep Q-Network). It uses
# Experience Replay and slow-learning target networks from DQN, and it
# is based on DPG, which can operate over continuous space actions.
# This example closely follows this paper: Continuous control with deep
# reinforcement learning.
# Problem.
# We are trying to solved the classic Inverted Pendulum control
# problem. In this setting, we can only take two actions: swing left or
# swing right. What makes this problem challenging for Q-Learning
# algorithms is that actions are continuous instead of being discrete.
# That is, instead of using two discrete actions like -1 or +1, we have
# to select from infinite actions ranging from -2 to +2.
# Quick Theory.
# Just like the Actor-Critic method, we have two networks:
# 1) Actor: It proposes an action given a state.
# 2) Critic: It predicts if the action is good (positive value) or bad
# (negative value) given a state and an action.
# DDPG uses two more techniques not present in the original DQN:
# First, it uses two target networks. Why? Because it adds stability to
# training. In short, we are learning from estimated targets and Target
# networks are updated slowly, hence keeping our estimated targets
# stable. Conceptually, this is like saying, "I have an idea of how to
# play this well, I'm going to try it out for a bit until I find
# something better", as opposed to saying "I'm going to re-learn how to
# play this entire game after every move".
# Second, it uses Experience Replay. We store a list of tuples (state,
# action, reward, next_state), and instead of only from recent
# experience, we learn from sampling all of our experience accumulated
# so far. 
# Source: https://keras.io/examples/rl/ddpg_pendulum/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def main():
	# Use OpenAIGym to create the environment. Use the upper_bound
	# parameter to scale actions later.
	problem = "Pendulum-v0"
	env = gym.make(problem)

	num_states = env.observation_space.shape[0]
	print("Size of State Space -> {}".format(num_states))
	num_actions = env.action_space.shape[0]
	print("Size of State Space -> {}".format(num_actions))

	upper_bound = env.action_space.high[0]
	lower_bound = env.action_space.low[0]

	print("Max Value of Action -> {}".format(upper_bound))
	print("Min Value of Action -> {}".format(lower_bound))

	# Training hyperparameters.
	std_dev = 0.2
	ou_noise = OUActionNoise(
		mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1)
	)

	actor_model = get_actor(num_states, upper_bound)
	critic_model = get_critic(num_states, num_actions)

	target_actor = get_actor(num_states, upper_bound)
	target_critic = get_critic(num_states, num_actions)

	# Making the weights equal initially.
	target_actor.set_weights(actor_model.get_weights())
	target_critic.set_weights(critic_model.get_weights())

	# Learning rate for actor-critic models.
	critic_lr = 0.002
	actor_lr = 0.001

	critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
	actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

	total_episodes = 100

	# Discount factor for future rewards.
	gamma = 0.99

	# Used to update target networks.
	tau = 0.005

	buffer = Buffer(
		num_states, num_actions, buffer_capacity=500000, batch_size=64
	)

	# Implement main training loop and iterate over episodes. Sample
	# actions using policy() and train with learn() at each time step,
	# along with updating the target networks at rate tau.
	# To store reward history of each episode.
	ep_reward_list = []

	# To store average reward history of last few episodes.
	avg_reward_list = []

	# Takes about 4 minutes to train.
	for ep in range(total_episodes):
		prev_state = env.reset()
		episodic_reward = 0

		while True:
			# Uncomment this to see the actor in action but not in a
			# python notebook.
			# env.render()

			tf_prev_state = tf.expand_dims(
				tf.convert_to_tensor(prev_state), 0
			)

			action = policy(
				tf_prev_state, ou_noise, lower_bound, upper_bound, actor_model
			)

			# Recieve state and reward from environment.
			state, reward, done, info = env.step(action)

			buffer.record((prev_state, action, reward, state))
			episodic_reward += reward

			buffer.learn(
				target_actor, target_critic, actor_model, critic_model, 
				actor_optimizer, critic_optimizer, gamma
			)
			update_target(target_actor.variables, actor_model.variables, tau)
			update_target(target_critic.variables, critic_model.variables, tau)

			# End this episode when "done" is True.
			if done:
				break

			prev_state = state

		ep_reward_list.append(episodic_reward)

		# Mean of last 40 episodes.
		avg_reward = np.mean(ep_reward_list[-40:])
		print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
		avg_reward_list.append(avg_reward)

	# Plotting graph. Episodes vs Avg Rewards.
	#'''
	plt.plot(avg_reward_list)
	plt.xlabel("Episode")
	plt.ylabel("Avg. Episodic Reward")
	plt.show()
	#'''

	# if training proceeds correctly, the average episodic reward will
	# increase with time. Feel free to try different learning rates,
	# tau values, and architectures for the Actor and Critic networks.
	# The Inverted Pendulum problem has low complexity, but DDPG works
	# great on many other problems. Another great environment to try
	# this on is LunarLandingContinuous-v2, but it will take more
	# episodes to obtain good results.

	# Save the weights.
	actor_model.save_weights("pendulum_actor.h5")
	critic_model.save_weights("pendulum_critic.h5")

	target_actor.save_weights("pendulum_target_actor.h5")
	target_critic.save_weights("pendulum_target_critic.h5")

	# Exit the program.
	exit(0)


# To implement better exploration by the Actor network, use noisy
# perturbations, specifically an Prnstein-Uhlenbeck process for
# generating noice, as described in the paper. it samples noise from a
# correlated normal distribution.
class OUActionNoise:
	def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
		self.theta = theta
		self.mean = mean
		self.std_dev = std_deviation
		self.dt = dt
		self.x_initial = x_initial
		self.reset()


	def __call__(self):
		# Formula taken from https://www.wikipedia.org/wiki/Ornstein-
		# Uhlenbeck_process.
		x = (
			self.x_prev
			+ self.theta * (self.mean - self.x_prev) * self.dt
			+ self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
		)
		# Store x into x_prev. Make next noise dependent on current
		# one.
		self.x_prev = x
		return x


	def reset(self):
		if self.x_initial is not None:
			self.x_prev = self.x_initial
		else:
			self.x_prev = np.zeros_like(self.mean)


# The Buffer class implements Experience Replay.
# Critic loss - Mean Squared Error of y - Q(s, a) where y is the
# expected return as seen by the Target network, and Q(s, a) is action
# value predicted by the Critic network. y is a moving target that the
# critic model tries to achieve; make this target stable by updating
# the Target model slowly.
# Actor loss - This is computed using the mean of the value given by
# the critic network for the actions taken by the actor network. Seek
# to maximize this quantity.
# Hence, update the actor network so that it produces actions that get
# the maximum predicted value as seen by the critic, for a given state.
class Buffer:
	def __init__(self, num_states, num_actions, buffer_capacity=100000, 
			batch_size=64):
		# Number of "experiences" to store at max.
		self.buffer_capacity = buffer_capacity

		# Num of tuples to train on.
		self.batch_size = batch_size

		# this tells the num of times record() was called.
		self.buffer_counter = 0

		# Instead of list of tuples as the exp.replay concept goes, use
		# different np.arrays for each tuple element.
		self.state_buffer = np.zeros((self.buffer_capacity, num_states))
		self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
		self.reward_buffer = np.zeros((self.buffer_capacity, 1))
		self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))


	# Takes (s, a, r, s') observation tuple as input.
	def record(self, obs_tuple):
		# Set index to zero if buffer capacity is exceeded, replacing
		# old records.
		index = self.buffer_counter % self.buffer_capacity

		self.state_buffer[index] = obs_tuple[0]
		self.action_buffer[index] = obs_tuple[1]
		self.reward_buffer[index] = obs_tuple[2]
		self.next_state_buffer[index] = obs_tuple[3]

		self.buffer_counter += 1


	# Eager execution is turned on by default in Tensorflow 2.
	# Decorating with tf.function allows Tensorflow to build a static
	# graph out of the logic and computations in our function. This
	# provides a large speed up for blocks fo code that contain many
	# small Tensorflow operations such as this one.
	@tf.function
	def update(self, state_batch, action_batch, reward_batch, next_state_batch,
			target_actor, target_critic, actor_model, critic_model, actor_optimizer, 
			critic_optimizer, gamma):
		# Training and updating Actor & Critic networks. See pseudo
		# code.
		with tf.GradientTape() as tape:
			target_actions = target_actor(next_state_batch, training=True)
			y = reward_batch + gamma * target_critic(
				[next_state_batch, target_actions], training=True
			)
			critic_value = critic_model([state_batch, action_batch], training=True)
			critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

		critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
		critic_optimizer.apply_gradients(
			zip(critic_grad, critic_model.trainable_variables)
		)

		with tf.GradientTape() as tape:
			actions = actor_model(state_batch, training=True)
			critic_value = critic_model([state_batch, actions], training=True)

			# Used "-value" to maximize the value given by the critic
			# for actions.
			actor_loss = -tf.math.reduce_mean(critic_value)

		actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
		actor_optimizer.apply_gradients(
			zip(actor_grad, actor_model.trainable_variables)
		)


	# Compute the loss and update parameters.
	def learn(self, target_actor, target_critic, actor_model, critic_model, 
			actor_optimizer, critic_optimizer, gamma):
		# Get sampling range.
		record_range = min(self.buffer_counter, self.buffer_capacity)

		# Randomly sample indices.
		batch_indices = np.random.choice(record_range, self.batch_size)

		# Convert to tensors.
		state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
		action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
		reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

		self.update(
			state_batch, action_batch, reward_batch, next_state_batch,
			target_actor, target_critic, actor_model, critic_model, 
			actor_optimizer, critic_optimizer, gamma)


# This updates target parameters slowly. Based on rate "tau", which is
# much less than one.
@tf.function
def update_target(target_weights, weights, tau):
	for (a, b) in zip(target_weights, weights):
		a.assign(b * tau + a * (1 - tau))


# Define the Actor and Critic networks. These are basic dense models
# with ReLU activation. Note: Need the initialization for last layer of
# the Actor to be between -0.003 and 0.003 as this prevents from
# getting 1 or -1 output values in the initial stages, which would
# squash gradients to zero, as the tanh function is used.
def get_actor(num_states, upper_bound):
	# Initialize weights between -3e-3 and 3-e3.
	last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

	inputs = layers.Input(shape=(num_states,))
	out = layers.Dense(256, activation="relu")(inputs)
	out = layers.Dense(256, activation="relu")(out)
	outputs = layers.Dense(
		1, activation="tanh", kernel_initializer=last_init
	)(out)

	# Upper bound is2.0 for Pendulum.
	outputs = outputs * upper_bound
	model = tf.keras.Model(inputs, outputs)
	return model


def get_critic(num_states, num_actions):
	# State as input.
	state_input = layers.Input(shape=(num_states))
	state_out = layers.Dense(16, activation="relu")(state_input)
	state_out = layers.Dense(32, activation="relu")(state_out)

	# Action as input.
	action_input = layers.Input(shape=(num_actions))
	action_out = layers.Dense(32, activation="relu")(action_input)

	# Both are passed through separate layer before concatenating.
	concat = layers.Concatenate()([state_out, action_out])

	out = layers.Dense(256, activation="relu")(concat)
	out = layers.Dense(256, activation="relu")(out)
	outputs = layers.Dense(1)(out)

	# Outputs single value for given state-action.
	model = tf.keras.Model([state_input, action_input], outputs)
	return model


# policy() returns an action sampled from the Actor network plus some
# noise for exploration.
def policy(state, noise_object, lower_bound, upper_bound, actor_model):
	sampled_actions = tf.squeeze(actor_model(state))
	noise = noise_object()

	# Adding noise to action.
	sampled_actions = sampled_actions.numpy() + noise

	# Make sure action is within bounds.
	legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
	return [np.squeeze(legal_action)]


if __name__ == '__main__':
	main()