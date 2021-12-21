# keras-tuner-example.py
# KerasTuner
# KerasTuner is an easy-to-use, scalable hyperparameter optimization
# framework that solves the pain points of hyperparameter search.
# Easily configure your search space with a define-by-run syntax, then
# leverage one of the available search algorithms to find the best
# hyperparameter values for your models. KerasTuner comes with Bayesian
# Optimization, Hyperband, and Random Search algorithms built-in, and
# is also designed to be easy for researchers to extend in order to
# experiment with new search algorithms.
# Quick Links
# Getting started with KerasTuner
# KerasTuner developer guides
# KerasTuner API reference
# Installation
# KerasTuner requirest Python 3.6+ and Tensorflow 2.0+
# Source: https://keras.io/keras_tuner/
# Tensorflow 2.4
# Python 3.7
# Windows/MacOS/Linux


import keras_tuner as kt
from tensorflow import keras


def main():
	# Quick Introduction
	# Write a function that creates and returns a Keras model. Use the
	# hp argument to define the hyperparameters during model creation.
	def build_model(hp):
		model = keras.Sequential()
		model.add(
			keras.layers.Dense(
				hp.Choice("units", [8, 16, 32]), activation="relu"
			)
		)
		model.add(keras.layers.Dense(1, activation="relu"))
		model.compile(loss="mse")
		return model

	# Initialize a tuner (here, RandomSearch). Use objective to specify
	# the objective to select the best models, and use max_trials to
	# specify the number of different models to try.
	tuner = kt.RandomSearch(
		build_model, objective="val_loss", max_trials=5
	)

	# Start the search and get the best model.
	tuner.search(
		x_train, y_train, epochs=5, validation_data=(x_val, y_val)
	)
	best_model = tuner.get_best_models()[0]

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()