# movie_rec_retrieval.py
# Build and train a two tower model using the Movielens dataset.
# Source: https://www.tensorflow.org/recommenders/examples/basic_retrieval
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import os
import pprint
import tempfile
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


def main():
	# Intro
	# Real-world recommender systems are often composed of two stages:
	# 1) The retrieval stage is responsible for selecting an initial
	#	set of hundreds of candidates from all possible candidates. The
	#	main objective of this model is to efficiently weed out all
	#	candidates that the user is not interested in. Because the
	#	retrieval model may be dealing with millions of candidates, it
	#	has to be computationally efficient.
	# 2) The ranking stage takes the outputs of the retrieval model and
	#	fine-tunes them to select the bes possible handful of 
	#	recommendations. Its task is to narrow down the set of items
	#	the user may be interested in to a shortlist of likely 
	#	candidates.
	# This tutorial is going to focus on the retrieval stage. The
	# ranking stage is built here
	# (https://www.tensorflow.org/recommenders/examples/basic_ranking).
	# Retrieval models are often composed of two sub-models:
	# 1) A query model computing the query representation (normally a
	#	fixed-dimensionality embedding vector) using query features.
	# 2) A candidate model computing the candidate representation (an
	#	equally-sized vector) using the candidate features.
	# This tutorial will build and train such a two-twower model using
	# the Movielens dataset.
	# In this tutorial, we will:
	# 1) Get the data and split it into a training and test set.
	# 2) Implement a retrieval model.
	# 3) Fit and evaluate it.
	# 4) Export it for efficient serving by building an approximate
	#	nearest neighbor (ANN) index.


	# The dataset
	# The Movielens dataset is a classic dataset from the GroupLens
	# research group at the University of Minnesota. It contains a set
	# of ratings given to movies by a set of users, and is a workhorse
	# of recommender system research.
	# The data can be treated in two ways:
	# 1) It can be interpreted as expressing which movies the users
	#	watched (and rated), and which they did not. This is a form of
	#	implicit feedback, where users' watches tell us which things
	#	they prefer to see and which they'd rather not see.
	# 2) It can also be seen as expressing how much the users liked the
	#	movies they did watch. This is a form of explicit feedback:
	#	given that a user watched a movie, we can tell roughly how
	#	much they liked by looking at the rating they have given.
	# As stated before, this tutorial is focusing on a retrieval
	# system: a model that predicts a set of movies from the catalogue
	# that the user is likely to watch. Often, implicit data is more
	# useful here, and so we are going to treat Movielens as an
	# implicit system. This means that every movie a user watched is a
	# positive example, and every movie they have not seen is an
	# implicit negative example.


	# Preparing the dataset
	# First, let's have a look at the data. We load the Movielens
	# dataset from Tensorflow Datasets. Loading movielens/100k_ratings
	# yields a tf.data.Dataset object containing the ratings data and
	# loading movielens/100_movies yields a tf.data.Dataset object
	# containing only the movies data. Note that since the Movielens
	# dataset does not have predefined splits, all data are under train
	# split.
	# Radings data.
	ratings = tfds.load("movielens/100k-ratings", split="train")
	# Features of all the available movies.
	movies = tfds.load("movielens/100k-movies", split="train")

	# The ratings dataset returns a dictionary of movie id, user id,
	# the assigned rating, timestamp, movie information, and user
	# information.
	for x in ratings.take(1).as_numpy_iterator():
		pprint.pprint(x)

	# The movies dataset contains the movie id, movie title, and data
	# one what genres it belongs to. Note that the genres are encoded
	# with integer labels.
	for x in movies.take(1).as_numpy_iterator():
		pprint.pprint(x)

	# In this example, we are going to focus on the ratings data. Other
	# tutorials explore how to use the movie information data as well
	# to improve the model quality. We keep only the user_id and
	# movie_title fields in the dataset.
	ratings = ratings.map(
		lambda x: {
			"movie_title": x["movie_title"],
			"user_id": x["user_id"],
		}
	)
	movies = movies.map(lambda x: x["movie_title"])

	# To fit and evaluate the model, we need to split it into a
	# training and evaluation set. In an industrial recommender system,
	# this would most likely be done by time: the data up to time
	# [Math Processing Error] would be used to predict interactions
	# after [Math Processing Error]. In this simple example, however,
	# let's use a random split, putting 80% of the ratings in the train
	# set, and 20% in the test set.
	tf.random.set_seed(42)
	shuffled = ratings.shuffle(
		100_000, seed=42, reshuffle_each_iteration=False
	)

	train = shuffled.take(80_000)
	test = shuffled.take(20_000)

	# Let's also figure out unique user ids and movie titles present in
	# the data. This is important because we need to be able to map the
	# raw values of our categorical features to embedding vectors in
	# our models. To do that, we need a vocabulary that maps a raw
	# feature value to an integer in a contiguous range: this allows us
	# to look up the corresponding embeddings.
	movie_titles = movies.batch(1_000)
	user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

	unique_movie_titles = np.unique(
		np.concatenate(list(movie_titles))
	)
	unique_user_ids = np.unique(np.concatenate(list(user_ids)))
	print(unique_movie_titles[:10])


	# Implementing a model
	# Choosing the architecture of our model is a key part of modeling.
	# Because we are building a two-tower retrieval model, we can build
	# each tower separately and then combine them in the final model.

	# The Query Tower:
	# Starting with the query tower, the first step is to decide on the
	# dimensionality of the query and candidate representations:
	embedding_dimension = 32

	# Higher values will correspond to models that may be more
	# accurate, but will also be slower to fit and more prone to 
	# overfitting.
	# The second is to define the model itself. Here, we're going to
	# use Keras preprocessing layers to first convert user ids to
	# integers, and then convert those to user embeddings via an
	# Embedding layer. Note that we use the list of unique user ids we
	# computed earlier as a vocabulary.
	user_model = tf.keras.Sequential([
		tf.keras.layers.StringLookup(
			vocabulary=unique_user_ids, mask_token=None
		),
		# We add an additional embedding to account for unknown tokens.
		tf.keras.layers.Embedding(
			len(unique_user_ids) + 1, embedding_dimension
		)
	])

	# A simple model like this corresponds exactly to a classic matrix
	# factorization approach. While defining a subclass of
	# tf.keras.Model for this simple model might be overkill, we can
	# easily extend it to an arbitrarily complex model using standard
	# Keras components, as long as we return an embedding_dimension
	# wide output at the end.

	# The Candidate Tower:
	# We can do the same thing with the candidate tower.
	movie_model = tf.keras.Sequential([
		tf.keras.layers.StringLookup(
			vocabulary=unique_movie_titles, mask_token=None
		),
		tf.keras.layers.Embedding(
			len(unique_movie_titles) + 1, embedding_dimension
		)
	])

	# Metrics:
	# In our training data we have positive (user, movie) pairs. To
	# figure out how good our model is, we need to compare the affinity
	# score that the model calculates for this pair to the scores of
	# all the other possible candidates: if the score for the positive
	# pair is higher than for all other candidates, our model is highly
	# accurate.
	# To do this, we can use the tfrs.metrics.FactorizedTopK metric.
	# The metric has one required argument: the dataset of candidates,
	# converted into embeddings via our movie model:
	metrics = tfrs.metrics.FactorizedTopK(
		candidates=movies.batch(128).map(movie_model)
	)

	# Loss:
	# The next component is the loss used to train our model. TFRS has
	# several loss layers and tasks to make this easy. In this
	# instance, we'll make use of the Retrieval task object: a
	# convenience wrapper that bundles together the loss function and
	# metric computation:
	task = tfrs.tasks.Retrieval(metrics=metrics)

	# The task itself is a Keras layer that takes the query and
	# candidate embeddings as arguments, and returns the computed loss:
	# we'll use that to implement the model's training loop.

	# The Full Model:
	# We can now put it all together into a model. TFRS exposes a base
	# model class (tfrs.models.Model) which streamlines building
	# models: all we need to do is set up the components in the
	# __init__ method, and implement the compute_loss method, taking
	# the raw features and returning a loss value.
	# The base model will then take care of creating the appropriate
	# training loop to fit our model.
	class MovielensModel(tfrs.Model):
		def __init__(self, user_model, movie_model):
			super().__init__()
			self.movie_model: tf.keras.Model = movie_model
			self.user_model:tf.keras.Model = user_model
			self.task: tf.keras.layers.Layer = task


		def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
			# We pick out the user features and pass them into the user
			# model.
			user_embeddings = self.user_model(features["user_id"])
			# And pick out the movie features and pass them into the
			# movie model, getting embeddings back.
			positive_movie_embeddings = self.movie_model(
				features["movie_title"]
			)

			# The task computes the loss and the metrics.
			return self.task(
				user_embeddings, positive_movie_embeddings
			)


	# The tfrs.Model base class is a simply convenience class: it
	# allows us to compute both training and test losses using the same
	# method. Under the hood, it's still a plain Keras model. You could
	# achieve the same functionality by inheriting from tf.keras.Model
	# and overriding the train_step and test_step functions (see the guide
	# https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
	# for details).
	class NoBaseClassMovielensModel(tf.keras.Model):
		def __init__(self, user_model, movie_model, task):
			super().__init__()
			self.movie_model: tf.keras.Model = movie_model
			self.user_model: tf.keras.Model = user_model
			self.task: tf.keras.Model = task


		def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
			# Set up a gradient tape to record gradients.
			with tf.GradientTape() as tape:
				# Loss computation.
				user_embeddings = self.user_model(features["user_id"])
				positive_movie_embeddings = self.movie_model(
					features["movie_title"]
				)
				loss = self.task(
					user_embeddings, positive_movie_embeddings
				)

				# Handle regularization losses as well.
				regularization_loss = sum(self.losses)

				total_loss = loss + regularization_loss

			gradients = tape.gradient(
				total_loss, self.trainable_variables
			)
			self.optimizer.apply_gradients(
				zip(gradients, self.trainable_variables)
			)

			metrics = {
				metrics.name: metrics.result() 
				for metric in self.metrics
			}
			metrics["loss"] = loss
			metrics["regularization_loss"] = regularization_loss
			metrics["total_loss"] = total_loss

			return metrics


		def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
			# Loss computation.
			user_embeddings = self.user_model(features["user_id"])
			positive_movie_embeddings = self.movie_model(
				features["movie_title"]
			)
			loss = self.task(
				user_embeddings, positive_movie_embeddings
			)

			# Handle regularization losses as well.
			regularization_loss = sum(self.losses)

			total_loss = loss + regularization_loss

			metrics = {
				metrics.name: metrics.result() 
				for metric in self.metrics
			}
			metrics["loss"] = loss
			metrics["regularization_loss"] = regularization_loss
			metrics["total_loss"] = total_loss

			return metrics

	# In these tutorials however, we stick to using the tfrs.Model base
	# class to keep our focus on modeling and abstract away some of the
	# boilerplate.


	# Fitting and evaluating
	# After defining the model, we can use standard Keras fitting and
	# evaluation routines to fit and evaluate the model. Let's first
	# instantiate the model.
	model = MovielensModel(user_model, movie_model)
	model.compile(
		optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)
	)

	# Then shuffle, batch, and cache the training and evaluation data.
	cached_train = train.shuffle(100_000).batch(8192).cache()
	cached_test = test.batch(4096).cache()

	# Then train the model:
	model.fit(cached_train, epochs=3)

	# If you wwant to monitor the training process with TensorBoard,
	# you can add a TensorBoard callback to fit() function and then
	# start TensorBoard using %tensorboard --logdir logs/fit. Please
	# refer to TensorBoard documentation for more details.
	# As the model trains, the loss is falling and a set of top-k
	# retrieval metrics is updated. These tell us whether the true
	# positive is in the top-k retrieved items from the entire
	# candidate set. For example, a top-5 categorical accuracy metric
	# of 0.2 would tell use that, on average, the true positive is in
	# the top 5 retrieved items 20% of the time.
	# Note that, in this example, we evaluate the metrics during
	# training as well as evaluation. Because this can be quite slow
	# with large candidate sets, it may be prudent to turn metric
	# calculation off in training, and only run it in evaluation.
	# Finally, we can evaluate our model on the test set:
	model.evaluate(cached_test, return_dict=True)

	# Test set performance is much words than training performance.
	# This is due to two factores:
	# 1) Our model is likely to perform better on the data that it has
	#	seen, simply because it can memorize it. This overfitting
	#	phenomenon is especially strong when models have many
	#	parameters. It can be mediated by model regularization and use
	#	of user and movie features that help the model generalize 
	#	better to unseen data.
	# 2) The model is re-recommending some of user's already watched 
	#	movies. These known-positive watches can crowd out test movies
	#	out of top k recommendations.
	# The second phenomenon can be tackled by excluding previously seen
	# movies from test recommendations. This approach is relatively
	# common in the recommender systems literature, but we don't follow
	# it in these tutorials. If not recommending past watches is
	# important, we should expect appropriately specified models to
	# learn this behaviour automatically from past user history and
	# contextual information. Additionally, it is often appropriate tp
	# recommend the same item multiple times (say, an evergreen TV
	# series or a regularly purchased item).


	# Making predictions
	# Now that we have a model, we would like to be able to make
	# predictions. We can use the tfrs.layers.factorized_top_k.BruteForce
	# layer to do this.
	# Create a model that takes in raw query features and
	index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
	# recommends movies out of the entire movies dataset.
	index.index_from_dataset(
		tf.data.Dataset.zip((
			movies.batch(100), movies.batch(100).map(model.movie_model)
		))
	)

	# Get recommendations.
	_, titles = index(tf.constant(["42"]))
	print(f"Recommendations for user 42: {titles[0, :3]}")

	# Of course, the BruteForce layer is going to be too slow to serve
	# a model with many possible candidates. The following sections
	# shows how to speed this up by using an approximate retrieval
	# index.


	# Model serving
	# After the model is trained, we need a way to deploy it.
	# In a two-tower retrieval model, serving has two components:
	# -> A serving query model, taking in features of the query and
	#	transforming them into a query embedding, and
	# -> A serving candidate model. This most often takes the form of
	#	an approximate nearest neighbors (ANN) index which allows fast
	#	approximate lookup of candidates in response to a query
	#	produced by the query model.
	# In TFRS, both components can be packaged into a single exportable
	# model, giving us a model that takes the raw user id and returns
	# the titles of top movies for that user. This is done via
	# exporting the model to a SavedModel format, which makes it
	# possible to serve using Tensorflow Serving.
	# To deploy a model like this, we simply export the BruteForce
	# layer we created above:

	# Export the query model.
	with tempfile.TemporaryDirectory() as tmp:
		path = os.path.join(tmp, "model")

		# Save the index.
		tf.saved_model.save(index, path)

		# Load it back; can also be done in Tensorflow Serving.
		loaded = tf.saved_model.load(path)

		# Pass a user id in, get top predicted movie titles back.
		scores, titles = loaded(["42"])
		print(f"Recommendations: {titles[0][:3]}")

	# We can also export an approximate retrieval index to speed up
	# predictions. This will make it possile to efficiently surface
	# recommendations from sets of tens of millions of candidates.
	# To do so, we can use the scann package. This is an optional
	# dependency of TFRS, and we installed it separately at the
	# beginning of this tutorial. Once installed, we can use the TFRS
	# SCaNN layer:
	scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
	scann_index.index_from_dataset(
		tf.data.Dataset.zip((
			movies.batch(100), movies.batch(100).map(model.movie_model)
		))
	)

	# This layer will perform approximate lookups: this makes retrieval
	# slightly less accurate, but orders of magnitude faster on large
	# candidate sets.
	# Get recommendations.
	_, titles = scann_index(tf.constant(["42"]))
	print(f"Recommendations for user 42: {titles[0, :3]}")

	# Exporting it for serving is as easy as exporting the BruteForce
	# layer:
	with tempfile.TemporaryDirectory() as tmp:
		path = os.path.join(tmp, "model")

		# Save the index.
		tf.saved_model.save(
			index, path, 
			options=tf.saved_model.SaveOptions(
				namespace_whitelist=["Scann"]
			)
		)

		# Load it back; can also be done in Tensorflow Serving.
		loaded = tf.saved_model.load(path)

		# Pass a user id in, get top predicted movie titles back.
		scores, titles = loaded(['42'])
		print(f"Recommendations: {titles[0][:3]}")

	# To learn more about using and tuning fast approximate retrieval
	# models, have a look at the efficient serving tutorial
	# https://www.tensorflow.org/recommenders/examples/efficient_serving.


	# Item-to-item recommendation
	# In this model, we created a user-movie model. However, for some
	# applications (for example, product detail pages) it's common to
	# perform item-to-item (for example, movie-to-movie or
	# product-to-product) recommendations.
	# Training models like this would follow the same pattern as shown
	# in this tutorial, but with different training data. Here, we had
	# a user and a movie tower, and used (user, movie) pairs to train
	# them. In an item-to-item model, we would have two item towers
	# (for the query and candidate item), and train the model using
	# (query item, candidate item) pairs. These could be constructed
	# from clicks on product detail pages.


	# Next steps
	# This concludes the retrieval tutorial. To expand on what has
	# been presented here, look at:
	# -> Learning multi-task models: jointly optimizing for ratings and
	#	clicks.
	# -> Using movie metadata: building a more complex movie model to
	#	alleviate cold-start.


	# Exit the program.
	exit()


if __name__ == '__main__':
	main()