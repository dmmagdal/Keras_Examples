# image_similarity_siamese_networks.py
# Training a Siamese Network to compare the similarity of images using
# a triplet loss function.
# A Siamese Network is a type of network architecture that contains two
# or more identical subnetworks used to generate feature vectors for
# each input and compare them. Siamese Networks can be applied to 
# different use cases, like detecting duplicates, finding anomalies,
# and face recognition. This example uses a Siamese Network with three
# identical subnetworks. Will provide three images to the model, where
# two of them will be similar (anchor and positive samples), and the
# third will be unrelated (a negative example). The goal is for the
# model to learn to estimate the similarity between images.
# For the network to learn, use a triplet loss function. There is an
# introduction to triplet loss in the FaceNet paper by Schroff et al.,
# 2015. In this example, the triplet loss function is defined as
# follows:
# L(A, P, N) = max(|f(A)-f(P)|^2 - |f(A) - f(N)|^2 + margin, 0)
# This example uses the Totally Looks Like dataset by Rosenfeld et al.,
# 2018.
# Source: https://keras.io/examples/vision/siamese_network/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from pathlib import Path
#'''
import matplotlib.pyplot as plt
#'''


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	global target_shape
	target_shape = (200, 200)

	# Load the dataset.
	# Load the Totally Looks Like dataset and unzip it in the ~/.keras
	# directory in the local environment.
	# The dataset consists of two separate files:
	# -> left.zip contains the images that will be used as the anchor.
	# -> right.zip contains the images that will be used as the 
	#	positive sample (an image that looks like the anchor).
	#cache_dir = Path(Path.home()) / ".keras"
	cache_dir = Path(Path.cwd()) / "keras"
	anchor_images_path = cache_dir / "left"
	positive_images_path = cache_dir / "right"

	# Preparing the data.
	# Use a tf.data pipeline to load the data and generate the triplets
	# that are needed to train the Siamese network. Will set up a
	# pipeline using a zipped list with anchor, positive, and negative
	# filenames as the source. The pipeline will load and preprocess
	# the corresponding images. Set up the data pipeline using a zipped
	# list with an anchor, positive, and negative image filename as the
	# source. The output of the pipeline contains the same triplet with
	# every image loaded and preprocessed.
	# Need to make sure both the anchor and positive images are loaded
	# in sorted order so that they can be matched together.
	anchor_images = sorted(
		[str(anchor_images_path / f) for f in os.listdir(anchor_images_path)]
	)

	positive_images = sorted(
		[str(positive_images_path / f) for f in os.listdir(positive_images_path)]
	)

	image_count = len(anchor_images)

	anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
	positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

	# To generate the list of negative images, randomize the list of
	# available images and concatenate them together.
	rng = np.random.RandomState(seed=42)
	rng.shuffle(anchor_images)
	rng.shuffle(positive_images)

	negative_images = anchor_images + positive_images
	np.random.RandomState(seed=32).shuffle(negative_images)

	negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
	negative_dataset = negative_dataset.shuffle(buffer_size=4096)

	dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
	dataset = dataset.shuffle(buffer_size=256)
	dataset = dataset.map(preprocess_triplets)

	# Split the dataset into train and validation.
	train_dataset = dataset.take(round(image_count * 0.8))
	val_dataset = dataset.skip(round(image_count * 0.8))

	train_dataset = train_dataset.batch(32, drop_remainder=False)
	train_dataset = train_dataset.prefetch(8)

	val_dataset = val_dataset.batch(32, drop_remainder=False)
	val_dataset = val_dataset.prefetch(8)

	# Take a look at a few samples of the triplets. Notice how the
	# first two images look alike while the third one is always
	# different.
	#'''
	visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])
	#'''

	# Setting up the embedding generator model.
	# The Siamese Network will generate embeddings for each of the
	# images of the triplet. To do this, use a ResNet50 model
	# pretrained on ImageNet and connect a few Dense layers to it so it
	# can learn to separate those embeddings.
	# Freeze the weights of all the layers of the model up until the
	# layer conv5_block1_out. This is important to avoid affecting the
	# weights that the model has already learned. Leave the bottom few
	# layers trainable, so that their weights can be fine-tuned during
	# training.
	base_cnn = resnet.ResNet50(
		weights="imagenet", input_shape=target_shape + (3,), include_top=False
	)

	flatten = layers.Flatten()(base_cnn.output)
	dense1 = layers.Dense(512, activation="relu")(flatten)
	dense1 = layers.BatchNormalization()(dense1)
	dense2 = layers.Dense(256, activation="relu")(dense1)
	dense2 = layers.BatchNormalization()(dense2)
	output = layers.Dense(256)(dense2)

	embedding = Model(base_cnn.input, output, name="Embedding")

	trainable = False
	for layer in base_cnn.layers:
		if layer.name == "conv5_block1_out":
			trainable = True
		layer.trainable = trainable

	# Setting up the Siamese Network model.
	# The Siamese network will receive each of the triplet images as an
	# input, generate the embeddings, and output the distance between
	# the anchor and the positive embedding, as well as the distance
	# between the anchor and the negative embedding. To compute the
	# distance, use a custom layer DistanceLayer that returns both
	# values as a tuple.
	anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
	positive_input = layers.Input(name="positive", shape=target_shape + (3,))
	negative_input = layers.Input(name="negative", shape=target_shape + (3,))

	distances = DistanceLayer()(
		embedding(resnet.preprocess_input(anchor_input)),
		embedding(resnet.preprocess_input(positive_input)),
		embedding(resnet.preprocess_input(negative_input)),
	)

	siamese_network = Model(
		inputs=[anchor_input, positive_input, negative_input], outputs=distances
	)

	# Training.
	siamese_model = SiameseModel(siamese_network)
	siamese_model.compile(optimizer=optimizers.Adam(0.0001))
	siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

	# Inspecting what the network has learned.
	# Use cosine similarity to measure similarity between embeddings.
	# Pick a sample from the dataset to check the similarity between
	# the embeddings generated for each image.
	sample = next(iter(train_dataset))
	#'''
	visualize(*sample)
	#'''

	anchor, positive, negative = sample
	anchor_embedding, positive_embedding, negative_embedding = (
		embedding(resnet.preprocess_input(anchor)),
		embedding(resnet.preprocess_input(positive)),
		embedding(resnet.preprocess_input(negative)),
	)

	# Compute the cosing similarity between the anchor and positive
	# images and compare it with the similarity between anchor and the
	# negative images. Expect the similarity between the anchor and
	# positive images to be larger than the similarity between the
	# anchor and the negative images.
	cosine_similarity = metrics.CosineSimilarity()

	positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
	print("Positive similarity:", positive_similarity.numpy())

	negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
	print("Negative similarity:", negative_similarity.numpy())

	# Exit the program.
	exit(0)


def preprocess_image(filename):
	# Load the specified file as a JPEG image, preprocess it and resize
	# it to the target shape.
	image_string = tf.io.read_file(filename)
	image = tf.image.decode_jpeg(image_string, channels=3)
	image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.image.resize(image, target_shape)
	return image


def preprocess_triplets(anchor, positive, negative):
	# Given the filenames corresponding to the three images, load and
	# preprocess them.
	return (
		preprocess_image(anchor), preprocess_image(positive), preprocess_image(negative)
	)


#'''
def visualize(anchor, positive, negative):
	# Visualize a few triplets from the supplied batches.
	def show(ax, image):
		ax.imshow(image)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

	fig = plt.figure(figsize(9, 9))

	axs = fig.subplots(3, 3)
	for i in range(3):
		show(axs[i, 0], anchor[i])
		show(axs[i, 1], positive[i])
		show(axs[i, 2], negative[i])
#'''


class DistanceLayer(layers.Layer):
	# This layer is responsible for computing the distance between the
	# anchor embedding and the positive embedding, and the anchor
	# embedding and the negative embedding.
	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	def call(self, anchor, positive, negative):
		ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
		an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
		return (ap_distance, an_distance)


# Putting everything together.
# Now implement a model with custom training loop so it can compute the
# triplet loss using the three embeddings produced by the Siamese
# network. Create a Mean metric instane to track the loss of the
# training process.
class SiameseModel(Model):
	# The Siamese Network model with a custom training and testing
	# loops. Computes the triplet loss usng the three embeddings
	# produced by the Siamese Network.
	# The triplet loss is defined as:
	# L(A, P, N) = max(|f(A)-f(P)|^2 - |f(A) - f(N)|^2 + margin, 0)
	def __init__(self, siamese_network, margin=0.5):
		super(SiameseModel, self).__init__()
		self.siamese_network = siamese_network
		self.margin = margin
		self.loss_tracker = metrics.Mean(name="loss")


	def call(self, inputs):
		return self.siamese_network(inputs)


	def train_step(self, data):
		# GradientTape is a context manager that records every
		# operation that occurs inside. It is used here to compute the
		# loss so it can get the gradients and apply them using the
		# optimizer specified in compile().
		with tf.GradientTape() as tape:
			loss = self._compute_loss(data)

		# Storing the gradients of the loss function with respect to
		# the weights/parameters.
		gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

		# Applying the gradients on the model using the specified
		# optimizer.
		self.optimizer.apply_gradients(
			zip(gradients, self.siamese_network.trainable_weights)
		)

		# Update and return the training loss metric.
		self.loss_tracker.update_state(loss)
		return {"loss": self.loss_tracker.result()}


	def test_step(self, data):
		loss = self._compute_loss(data)

		# update and return the loss metric.
		self.loss_tracker.update_state(loss)
		return {"loss": self.loss_tracker.result()}


	def _compute_loss(self, data):
		# The output of the network is a tuple containing the distances
		# between the anchor and the positive example, and the anchor
		# and the negative example.
		ap_distance, an_distance = self.siamese_network(data)

		# Computing the triplet loss by subtracting both distances and
		# making sure it doesnt get a negative result.
		loss = ap_distance - an_distance
		loss = tf.maximum(loss + self.margin, 0.0)
		return loss


	@property
	def metrics(self):
		# List metrics here so the reset_states() can be called
		# automatically.
		return [self.loss_tracker]


if __name__ == '__main__':
	main()