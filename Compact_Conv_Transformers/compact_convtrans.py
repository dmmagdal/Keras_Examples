# compact_convtrans.py
# Compact convolutional transformers for efficient image
# classification.
# As discussed in the Visiton Transformers (ViT) paper, a trnasformer-
# based architecture for vision typically requires a larger dataset
# than usual, as well as a longer pre-training schedule. ImageNet-1k
# (which has about a million images) is considered to fall under the
# medium-sized data regime with respect to ViTs. This is primarily
# because, unlike CNs, ViTs (or a typical transformer-based
# architecture) do not have well-informed inductive biases (such as
# convolutions for processing images). This begs the question: cant we
# combine the benefites of convolution and the benefits of transformers
# in a single network architecture? These benefits include parameter-
# efficiency, and self-attention to process long-range and global
# dependencies (interactions between different regions in an image).
# In Escaping the Big Data Paradigm with Compact Transformers, Hassani
# et al. present an approach for doing exactly this. They proposed the
# Compact Convolutional Transformer (CCT) architecture. This example
# workds to implement a CCT and see how well it performs on the
# CIFAR-10 dataset.
# If you are unfamiliar with the concept of self-attention or 
# transformers, you can read this chapter from Francios Chollet's book
# Deep Learning with Python. This example uses code snippets from
# another example, Image classification with Vision Transformers.
# Source: https://keras.io/examples/vision/cct/
# Tensorflow 2.7
# Python 3.7
# Windows/MacOS/Linux


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
#import matplotlib.pyplot as plt


def main():
	# Hyperparameters and constraints
	positional_emb = True
	conv_layers = 2
	projection_dim = 128

	num_heads = 2
	transformer_units = [
		projection_dim,
		projection_dim,
	]
	transformer_layers = 2
	stochastic_depth_rate = 0.1

	learning_rate = 0.001
	weight_decay = 0.0001
	batch_size = 128
	num_epochs = 30
	image_size = 32

	# Load CIFAR-10 dataset
	num_classes = 10
	input_shape = (32, 32, 3)

	(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
	print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

	# The CCT tokenizer
	# The first recipe introduced by the CCT authors is the tokenizer
	# for processing the images. In a standard ViT, images are
	# organized into uniform non-overlapping patches. This eliminates
	# the boundary-level information present in between different
	# patches. This is important for a neural network to effectively
	# exploit the locality information. The figure below presents an
	# illustration of how images are organized into patches.
	# We already know that convolutions are quite good at exploiting
	# locality information. So, based on this, the authors introduced
	# an all-convolutional mini-network to produce image patches.
	class CCTTokenizer(layers.Layer):
		def __init__(self, kernel_size=3, stride=1, padding=1,
				pooling_kernel_size=3, pooling_stride=2,
				num_conv_layers=conv_layers, 
				num_output_channels=[64, 128], 
				positional_emb=positional_emb, **kwargs):
			super(CCTTokenizer, self).__init__(**kwargs)

			# This is the tokenizer.
			self.conv_model = keras.Sequential()
			for i in range(num_conv_layers):
				self.conv_model.add(
					layers.Conv2D(
						num_output_channels[i], kernel_size, stride,
						padding="valid", use_bias=False, 
						activation="relu", kernel_initializer="he_normal",
					)
				)
				self.conv_model.add(layers.ZeroPadding2D(padding))
				self.conv_model.add(
					layers.MaxPooling2D(
						pooling_kernel_size, pooling_stride, "same"
					)
				)
			self.positional_emb = positional_emb


		def call(self, images):
			outputs = self.conv_model(images)

			# After passing the images through the mini-network the
			# spatial dimensions are flattened to form sequences.
			reshaped = tf.reshape(
				outputs,
				(
					-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], 
					tf.shape(outputs)[-1]
				)
			)
			return reshaped


		def positional_embedding(self, image_size):
			# Positional embeddings are optional in CCT. Here, we
			# calculate the number of sequences and initialize an
			# 'Embedding' layer to computer the positional embeddings
			# later.
			if self.positional_emb:
				dummy_inputs = tf.ones((1, image_size, image_size, 3))
				dummy_outputs = self.call(dummy_inputs)
				sequence_length = tf.shape(dummy_outputs)[1]
				projection_dim = tf.shape(dummy_outputs)[-1]

				embed_layer = layers.Embedding(
					input_dim=sequence_length, output_dim=projection_dim
				)
				return embed_layer, sequence_length
			else:
				return None


	# Stochastic depth for regularization
	# Stochastic depth is a regularization technique that randomly
	# drops a set of layers. During inference, the layers are kept as
	# they are. It is very much similar to Dropout but only that it
	# operates on a block of layers rather than individual nodes
	# present inside a layer. In CCT, stochastic depth is used just
	# before the residual blocks of a transformer encoder.
	# Referred from: github.com:rwightman/pytorch-image-models.
	class StochasticDepth(layers.Layer):
		def __init__(self, drop_prop, **kwargs):
			super(StochasticDepth, self).__init__(**kwargs)
			self.drop_prob = drop_prop


		def call(self, x, training=None):
			if training:
				keep_prob = 1 - self.drop_prob
				shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
				random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
				random_tensor = tf.floor(random_tensor)
				return (x / keep_prob) * random_tensor
			return x


	# MLP for the transformer encoder
	def mlp(x, hidden_units, dropout_rate):
		for units in hidden_units:
			x = layers.Dense(units, activation=tf.nn.gelu)(x)
			x = layers.Dropout(dropout_rate)(x)
		return x


	# Data augmentation
	# In the original paper, the authors use AutoAugment to induce
	# stronger regularization. For this example, use the standard
	# geometric augmentations like random cropping and flipping.
	# Note the rescaling layer. These layers have pre-defined inference
	# behavior.
	data_augmentation = keras.Sequential(
		[
			layers.Rescaling(scale=1.0 / 255),
			layers.RandomCrop(image_size, image_size),
			layers.RandomFlip('horizontal'),
		],
		name="data_augmentation",
	)

	# The final CCT model
	# Another recipe introduced in CCT is attention pooling or sequence
	# pooling. In ViT, only the feature map corresponding to the class
	# token is pooled and is then used for the subsequent
	# classification task (or any other downstream task). In CCT,
	# outputs from the transformers encoder are weighted and then
	# passed on to the final task specific layer (in this example, we
	# do classification).
	def create_cct_model(image_size=image_size, input_shape=input_shape,
			num_heads=num_heads, projection_dim=projection_dim,
			transformer_units=transformer_units):
		inputs = layers.Input(input_shape)

		# Augment data.
		augmented = data_augmentation(inputs)

		# Encode patches.
		cct_tokenizer = CCTTokenizer()
		encoded_patches = cct_tokenizer(augmented)

		# Apply positional embedding.
		if positional_emb:
			pos_embed, seq_length = cct_tokenizer.positional_embedding(
				image_size
			)
			positions = tf.range(start=0, limit=seq_length, delta=1)
			position_embeddings = pos_embed(positions)
			encoded_patches += position_embeddings

		# Calculate Stochastic Depth probabilities.
		dpr = [
			x for x in np.linspace(
				0, stochastic_depth_rate, transformer_layers
			)
		]

		# Create multiple layers of the transformer block.
		for i in range(transformer_layers):
			# Layer normalization 1.
			x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

			# Create a multi-head attention layer.
			attention_output = layers.MultiHeadAttention(
				num_heads=num_heads, key_dim=projection_dim, dropout=0.1
			)(x1, x1)

			# Skip connection 1.
			attention_output = StochasticDepth(dpr[i])(attention_output)
			x2 = layers.Add()([attention_output, encoded_patches])

			# Layer normalization 2.
			x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

			# MLP.
			x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

			# Skip connection 2.
			x3 = StochasticDepth(dpr[i])(x3)
			encoded_patches = layers.Add()([x3, x2])

		# Apply sequence pooling.
		representation = layers.LayerNormalization(epsilon=1e-5)(
			encoded_patches
		)
		attention_weights = tf.nn.softmax(
			layers.Dense(1)(representation), axis=1
		)
		weighted_representation = tf.matmul(
			attention_weights, representation, transpose_a=True
		)
		weighted_representation = tf.squeeze(weighted_representation, -2)

		# Classify outputs.
		logits = layers.Dense(num_classes)(weighted_representation)

		# Create the keras model.
		model = keras.Model(inputs=inputs, outputs=logits)
		return model


	# Model training and evaluation
	def run_experiment(model):
		optimizer = tfa.optimizers.AdamW(
			learning_rate=0.001, weight_decay=0.0001
		)

		model.compile(
			optimizer=optimizer,
			loss=keras.losses.CategoricalCrossentropy(
				from_logits=True, label_smoothing=0.1
			),
			metrics=[
				keras.metrics.CategoricalAccuracy(name="accuracy"),
				keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy")
			],
		)

		checkpoint_filepath = "/tmp/checkpoint"
		checkpoint_callback = keras.callbacks.ModelCheckpoint(
			checkpoint_filepath, monitor="val_accuracy",
			save_best_only=True, save_weights_only=True,
		)

		history = model.fit(
			x=x_train, y=y_train, batch_size=batch_size,
			epochs=num_epochs, validation_split=0.1,
			callbacks=[checkpoint_callback],
		)

		model.load_weights(checkpoint_filepath)
		_, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
		print(f"Test accuracy: {round(accuracy * 100, 2)}%")
		print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

		return history


	cct_model = create_cct_model()
	history = run_experiment(cct_model)

	# Now visualize the training progress of the model.
	'''
	plt.plot(history.history["loss"], label="train_loss")
	plt.plot(history.history["val_loss"], label="val_loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.title("Train and Validation Losses Over Epochs", fontsize=14)
	plt.legend()
	plt.grid()
	plt.show()
	'''

	# The CCT model trained above has just 0.4 million parameters, and
	# it gets to ~78% top-1 accuracy within 30 epochs. The plot above
	# shows no signs of overfitting as well. This means we can train
	# this network for longer (perhaps with a bit more regularization)
	# and may obtain even better performance. This performance can
	# further be improved by additional recipes like cosine decay 
	# learning rate schedule, other data augmentation techniques like
	# AutoAugment, MixUp or Cutmix. With these modifications, the
	# authors present 95.1% top-1 accuracy on the CIFAR-10 dataset. THe
	# authors also present a number of experiments to study how the
	# number of convolution blocks, transformer layers, etc. affect the
	# final performance of CCTs.
	# For a comparison, a ViT model takes about 4.7 million parameters
	# and 100 epochs of training to reach top-1 accuracy of 78.22% on
	# the CIFAR-10 dataset. You can refer to this notebook to know
	# about the experimental setup.
	# The authors also demonstrate the performance of Compact
	# Convolutional Transformers on NLP tasks and they report
	# competitive results there.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()