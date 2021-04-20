# actor_critic_learning.py
# Implement Actor Critic Method in CartPole environment.
# Actor Critic Method
# As an agent takes actions and moves through an environment, it learns
# to map the observed state of the environment to two possible outputs:
# 1) Recommended action: A probability value for each action in the
#	action space. The part of the agent responsible for this output is
#	called the actor.
# 2) Estimated rewards in the future: Sum of all rewards it expects to
#	receive in the future. The part of the agent responsible for this
#	output is the critic.
# Agent and Critic learn to perform their tasks, such that the
# recommended actions from the actor maximize the rewards.
# CartPole-V0
# A pole is attached to a cart placed on a frictionless track. The
# agent has to apply force to move the cart. It is rewarded for every
# time step the pole remains upright. The agent, therefore, must learn
# to keep the pole from falling over.
# Source: https://keras.io/examples/rl/actor_critic_cartpole/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
	# Configuration parameters for the whole setup.
	seed = 42
	gamma = 0.99 # Discount factor for past rewards.
	max_steps_per_episode = 1000
	env = gym.make("CartPole-v0") # Create the environment
	env.seed(seed)
	eps = np.finfo(np.float32).eps.item()

	# Implement Actor Critic network. This network learns two
	# functions:
	# 1) Actor: This takes as input the state of the environment and
	#	returns a probability value for each action in its action
	#	space.
	# 2) Critic: This takes as input the state of the environment and
	#	returns an estimate of total rewards in the future.
	# In this implementation, they share the initial layer.
	num_inputs = 4
	num_actions = 2
	num_hidden = 128

	inputs = layers.Input(shape=(num_inputs,))
	common = layers.Dense(num_hidden, activation="relu")(inputs)
	action = layers.Dense(num_actions, activation="softmax")(common)
	critic = layers.Dense(1)(common)

	model = keras.Model(inputs=inputs, outputs=[action, critic])

	# Train.
	optimizer = keras.optimizers.Adam(learning_rate=0.01)
	huber_loss = keras.losses.Huber()
	action_probs_history = []
	critic_value_history = []
	rewards_history = []
	running_reward = 0
	episode_count = 0

	# Run until solved.
	while True:
		state = env.reset()
		episode_reward = 0
		with tf.GradientTape() as tape:
			for timestep in range(1, max_steps_per_episode):
				# env.render(); Adding this line would show the
				# attempts of the agent in a pop up window.

				state = tf.convert_to_tensor(state)
				state = tf.expand_dims(state, 0)

				# Predict action probabilities and estimated future
				# rewards from environment state.
				action_probs, critic_value = model(state)
				critic_value_history.append(critic_value[0, 0])

				# Sample action probabilities and estimate future
				# rewards from environment state.
				action = np.random.choice(num_actions, p=np.squeeze(action_probs))
				action_probs_history.append(tf.math.log(action_probs[0, action]))

				# Apply the sampled action in the environment.
				state, reward, done, _ = env.step(action)
				rewards_history.append(reward)
				episode_reward += reward

				if done:
					break

			# Update running reward to check condition for solving.
			running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

			# Calculate expected value from rewards.
			# - At each timestep what was the total reward received
			#	after that timestep.
			# - Rewards in the past are discounted by multiplying them
			#	them with gamma.
			# - These are the labels for our critic.
			returns = []
			discounted_sum = 0
			for r in rewards_history[::-1]:
				discounted_sum = r + gamma * discounted_sum
				returns.insert(0, discounted_sum)

			# Normalize.
			returns = np.array(returns)
			returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
			returns = returns.tolist()

			# Calculating loss values to update the network.
			history = zip(action_probs_history, critic_value_history, returns)
			actor_losses = []
			critic_losses = []
			for log_prob, value, ret in history:
				# At this point in history, the critic estimated that
				# we would get a total reward = "value" in the future.
				# Take an action with log probability of "log_prob" and
				# ended up receiving a total reward = "ret". The actor
				# must be updated so that it predicts an action that
				# leads to high rewards (compared to critic's estimate)
				# with high probability.
				diff = ret - value
				actor_losses.append(-log_prob * diff) # Actor loss.

				# The critic must be updated so that it predicts a
				# better estimate of the future rewards.
				critic_losses.append(
					huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
				)

			# Backpropagation.
			loss_value = sum(actor_losses) + sum(critic_losses)
			grads = tape.gradient(loss_value, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))

			# Clear the lost and reward history.
			action_probs_history.clear()
			critic_value_history.clear()
			rewards_history.clear()

		# Log details.
		episode_count += 1
		if episode_count % 10 == 0:
			template = "running reward: {:.2f} at episode {}"
			print(template.format(running_reward, episode_count))

		if running_reward > 195:
			print("Solved at episode {}!".format(episode_count))
			break

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()