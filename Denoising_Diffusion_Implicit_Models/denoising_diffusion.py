# denoising_diffusion.py
# Create and train a minimal implementation of diffusion models, with
# modest compute requirements and reasonable performance.
# Introduction
# What are diffusion models?
# Recently, denoising diffusion models, including score-based
# generative models, gained popularity as a powerful class of
# generative models, that can rival even generative adversarial
# networks (GANs) in image synthesis quality. They tend to generate
# more diverse samples, while being stable to train and easy to scale.
# Recent large diffusion models, such as DALL-E 2 and Imagen, have
# shown incredible text-to-image generation capability. One of their
# drawbacks is however, that they are slower to sample from, because
# they cause multiple forward passes for generating an image.
# Diffusion refers to the process of turning a structured signal (an
# image) into noise step-by-step. By simulating diffusion, we can
# generate noisy images from our training images, and can train a
# neural network to try and denoise them. Using the training network we
# can simulate the opposite of diffusion, reverse diffusion, which is
# the process of an image emerging from noise.
# One-sentence summary: diffusion models are trained to denoise noisy
# images, and can generate images by iteratively denoising pure noise.
# Goal of this example
# This code example intends to be a minimal but feature-complete (with
# a generation quality metric) implementation of diffusion models, with
# modest compute requirements and reasonable performance. The
# implementation choices and hyperparameter tuning were done with these
# goals in mind.
# Since currently the literature of diffusion models is mathematically
# quite complext with multiple theoretical frameworks (score matching,
# differential equations, Markov chains) and sometimes even conflicting
# notations (see Appendix C.2), it can be daunting trying to understand
# them. This example's view of these models in this example will be
# that they learn to separate a noisy image into its image and Gaussian
# noise components.
# This example will make the effort to break down all long mathematical
# expressions into digestible pieces and give all variables explanatory
# names. There are also numerous links to relevant literature to help
# interested readers dive deeper into the topic, in the hope that this
# code example will become a good starting point for practitioners
# learning about diffusion models.
# The following implements a continuous time version of Denoising
# Diffusion Probabilistic Models (DDIMs) with deterministic sampling.
# Source: https://keras.io/examples/generative/ddim/
# Tensorflow 2.7
# Python 3.7
# Windows/MacOS/Linux


import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_addons as tfa


def main():
	# Hyperparameters
	# Data.
	dataset_name = "oxford_flowers102"
	dataset_repetitions = 5
	num_epochs = 1  # train for at least 50 epochs for good results
	image_size = 64

	# KID = Kernel Inception Distance, see related section.
	kid_image_size = 75
	kid_diffusion_steps = 5
	plot_diffusion_steps = 20

	# Sampling.
	min_signal_rate = 0.02
	max_signal_rate = 0.95

	# Architecture.
	embedding_dims = 32
	embedding_max_frequency = 1000.0
	widths = [32, 64, 96, 128]
	block_depth = 2

	# Optimization.
	batch_size = 64
	ema = 0.999
	learning_rate = 1e-3
	weight_decay = 1e-4


	# Data Pipeline
	# This example will use the Oxford Flowers 102 dataset for
	# generating images of flowers, which is a diverse natural dataset
	# containing around 8,000 images. Unfortunately the official splits
	# are imbalaced, as most of the images are contained in the test
	# split. We create new splits (80/20 train/test split) using the
	# Tensorflow Datasets slicing API. We apply center crops as
	# preprocessing, and repeat with the dataset multiple times (reason
	# given in the next section).
	def preprocess_image(data):
		# Center crop image.
		height = tf.shape(data["image"])[0]
		width = tf.shape(data["image"])[1]
		crop_size = tf.math.minimum(height, width)
		image = tf.image.crop_to_bounding_box(
			data["image"], 
			(height - crop_size) // 2, 
			(width - crop_size) // 2,
			crop_size, crop_size,
		)

		# Resize and clip. For image downsampling, it is important to
		# turn on antialiasing.
		image = tf.image.resize(
			image, size=[image_size, image_size], antialias=True
		)
		return tf.clip_by_value(image / 255.0, 0.0, 1.0)


	def prepare_dataset(split):
		# The validation dataset is shuffled as well, because data
		# order matters for the KID estimation.
		return (
			tfds.load(
				dataset_name, split=split, shuffle_files=True
			)
			.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
			.cache()
			.repeat(dataset_repetitions)
			.shuffle(10 * batch_size)
			.batch(batch_size, drop_remainder=True)
			.prefetch(buffer_size=tf.data.AUTOTUNE)
		)


	# Load dataset.
	train_dataset = prepare_dataset(
		"train[:80%]+validation[:80%]+test[:80%]"
	)
	val_dataset = prepare_dataset(
		"train[80%:]+validation[80%:]+test[80%:]"
	)


	# Kernel Inception Distance (KID)
	# Kernel Inception Distance (KID) is an image quality metric which
	# was proposed as a replacement for the popular Frechet Inception
	# Distance (FID). This example prefers KID to FID because it is
	# simpler to implement, can be estimated per-batch, and is
	# computationally lighter. More details here:
	# https://keras.io/examples/generative/gan_ada/#kernel-inception-
	# distance.
	# In this example, the images are evaluated at the minimal possible
	# resolution of the inception network (75x75 instead of 299x299),
	# and the metric is only measured on the validation set for
	# computational efficiency. This example also limits the number of
	# sampling steps at evaluation to 5 for the same reason.
	# Since the dataset is relatively small, we go over the train and
	# validation splits multiple times per epoch, because the KID
	# estimation is noisy and compute-intensive, so we want to evaluate
	# only after many iterations, but not for many iterations.
	class KID(keras.metrics.Metric):
		def __init__(self, name, **kwargs):
			super().__init__(name=name, **kwargs)

			# KID is estimated per batch and is averaged across
			# batches.
			self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

			# A pretrained InceptionV3 is used without its
			# classification layer. Transform the pixel values to the
			# 0-255 range, then use the same preprocessing as during
			# pretraining.
			self.encoder = keras.Sequential(
				[
					keras.Input(shape=(image_size, image_size, 3)),
					layers.Rescaling(255.0),
					layers.Resizing(
						height=kid_image_size, width=kid_image_size
					),
					layers.Lambda(
						keras.applications.inception_v3.preprocess_input
					),
					keras.applications.InceptionV3(
						include_top=False,
						input_shape=(
							kid_image_size, kid_image_size, 3
						),
						weights="imagenet",
					),
					layers.GlobalAveragePooling2D(),
				],
				name="inception_encoder"
			)


		def polynomial_kernel(self, features_1, features_2):
			feature_dimensions = tf.cast(
				tf.shape(features_1)[1], dtype=tf.float32
			)
			return (
				features_1 @ tf.transpose(features_2) /
				feature_dimensions + 1.0
			) ** 3.0


		def update_state(self, real_images, generated_images, 
				sample_weight=None):
			real_features = self.encoder(real_images, training=False)
			generated_features = self.encoder(
				generated_images, training=False
			)

			# Compute polynomial kernels using the two sets of
			# features.
			kernel_real = self.polynomial_kernel(
				real_features, real_features
			)
			kernel_generated = self.polynomial_kernel(
				generated_features, generated_features
			)
			kernel_cross = self.polynomial_kernel(
				real_features, generated_features
			)

			# Estimate the squared maximum mean discrepancy using the
			# average kernel values.
			batch_size = tf.shape(real_features)[0]
			batch_size_f = tf.cast(batch_size, dtype=tf.float32)
			mean_kernel_real = tf.math.reduce_sum(
				kernel_real * (1.0 - tf.eye(batch_size))
			) / (batch_size_f * (batch_size_f - 1.0))
			mean_kernel_generated = tf.math.reduce_sum(
				kernel_generated * (1.0 - tf.eye(batch_size))
			) / (batch_size_f * (batch_size_f - 1.0))
			mean_kernel_cross = tf.math.reduce_sum(kernel_cross)
			kid = mean_kernel_real + mean_kernel_generated - 2.0 *\
				mean_kernel_cross

			# Update the average KID estimate.
			self.kid_tracker.update_state(kid)


		def result(self):
			return self.kid_tracker.result()


		def reset_state(self):
			self.kid_tracker.reset_state()


	# Network Architecture
	# Here we specify the architecture of the neural network that we
	# will use for denoising. We build a U-Net with identical input and
	# output dimensions. U-Net is a popular semantic segmentation
	# architecture, whose main idea is that it progressively
	# downsamples and then upsamples its input image, and adds skip
	# connections between layers having the same resolution. These help
	# with gradient flow and avoid introducing a representation
	# bottleneck, unlike usual autoencoders. Based on this, one can
	# view diffusion models as denoising autoencoders without a
	# bottleneck.
	# The network takes two inputs, the noisy images and the variances
	# of their noise components. The latter is required since denoising
	# a signal requires different operations at different levels of
	# noise. We transform the noise variances using sinusoidal
	# embeddings, similarly to positional encodings used both in
	# transformers and NeRF. This helps the network to be highly
	# sensitive to the noise-level, which is crucial for good
	# performance. We implement sinusoidal embeddings using a lambda
	# layer.
	# Some other considerations:
	# -> We build the network using the Keras Functional API, and use
	#	closures to build blocks of layers in a consistent style.
	# -> Diffusion models embed the index of the timestep of the
	#	diffusion process instead of the noise variance, while score-
	#	based models (Table 1) usually use some function of the noise
	#	level. This example prefers the latter so that we can change
	#	the sampling schedule at inference time, without retraining the
	#	network.
	# -> Diffusion models input the embedding to each convolution block
	#	separately. We only input it at the start of the network for
	#	simplicity, which in my experience barely decreases 
	#	performance, because the skip and residual connections help the
	#	information propagate through the network properly.
	# -> In the literature, it is common to use attention layers at
	#	lower resolutions for better global coherence. This is omitted
	#	for simplicity.
	# -> We disable the learnable center and scale parameters of the
	#	batch normalization layers, since the following convolution
	#	layers make them redundant.
	# -> We initialize the last convolution's kernel to all zeros as a
	#	good practice, making the network predict only zeros after
	#	initialization, which is the mean of its targets. Ths will
	#	improve behavior at the start of training and make the mean
	#	squared error loss start at exactly 1.
	def sinusoidal_embedding(x):
		embedding_min_frequency = 1.0
		frequencies = tf.math.exp(
			tf.linspace(
				tf.math.log(embedding_min_frequency),
				tf.math.log(embedding_max_frequency),
				embedding_dims // 2,
			)
		)
		angular_speeds = 2.0 * math.pi * frequencies
		embeddings = tf.concat(
			[
				tf.math.sin(angular_speeds * x),
				tf.math.cos(angular_speeds * x)
			], axis=3
		)
		return embeddings


	'''
	def ResidualBlock(width):
		def apply(x):
			input_width = x.shape[3]
			if input_width == width:
				residual = x
			else:
				residual = layers.Conv2D(width, kernel_size=1)(x)
			x = layers.BatchNormalization(center=False, scale=False)(x)
			x = layers.Conv2D(
				width, kernel_size=3, padding="same",
				activation=keras.activations.swish
			)(x)
			x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
			x = layers.Add()([x, residual])

		return apply


	def DownBlock(width, block_depth):
		def apply(x):
			x, skips = x
			for _ in range(block_depth):
				x = ResidualBlock(width)(x)
				skips.append(x)
			x = layers.AveragePooling2D(pool_size=2)(x)
			return x

		return apply


	def UpBlock(width, block_depth):
		def apply(x):
			x, skips = x
			x = layers.UpSampling2D(
				size=2, interpolation="bilinear"
			)(x)
			for _ in range(block_depth):
				x = layers.Concatenate()([x, skips.pop()])
				x = ResidualBlock(x)
			return x

		return apply
	'''


	class ResidualBlock(layers.Layer):
		def __init__(self, width):
			super(ResidualBlock, self).__init__()

			self.width = width
			self.residual_conv = layers.Conv2D(
				width, kernel_size=1
			)
			self.bn = layers.BatchNormalization(
				center=False, scale=False
			)
			self.conv1 = layers.Conv2D(
				width, kernel_size=3, padding="same",
				activation=keras.activations.swish
			)
			self.conv2 = layers.Conv2D(
				width, kernel_size=3, padding="same"
			)
			self.add = layers.Add()


		def call(self, x):
			input_width = x.shape[3]
			if input_width == self.width:
				residual = x
			else:
				residual = self.residual_conv(x)
			x = self.bn(x)
			x = self.conv1(x)
			x = self.conv2(x)
			x = self.add([x, residual])
			return x


		def get_config(self):
			# return {
			# 	"width": self.width,
			# }
			config = super(ResidualBlock, self).get_config()
			config.update({
				"width": self.width,
			})
			return config


	class DownBlock(layers.Layer):
		def __init__(self, width, block_depth):
			super(DownBlock, self).__init__()

			self.width = width
			self.block_depth = block_depth
			self.resblocks = [
				ResidualBlock(width) for _ in range(block_depth)
			]
			self.avgpool2d = layers.AveragePooling2D(pool_size=2)


		def call(self, x):
			x, skips = x
			for i in range(self.block_depth):
				x = self.resblocks[i](x)
				skips.append(x)
			x = self.avgpool2d(x)
			return x, skips


		def get_config(self):
			# return {
			# 	"width": self.width,
			# 	"block_depth": self.block_depth,
			# }
			config = super(DownBlock, self).get_config()
			config.update({
				"width": self.width,
				"block_depth": self.block_depth,
			})
			return config


	class UpBlock(layers.Layer):
		def __init__(self, width, block_depth):
			super(UpBlock, self).__init__()

			self.width = width
			self.block_depth = block_depth
			self.upsample2d = layers.UpSampling2D(
				size=2, interpolation="bilinear"
			)
			self.resblocks = [
				ResidualBlock(width) for _ in range(block_depth)
			]


		def call(self, x):
			x, skips = x
			x = self.upsample2d(x)
			for i in range(self.block_depth):
				x = layers.Concatenate()([x, skips.pop()])
				x = self.resblocks[i](x)
			return x, skips


		def get_config(self):
			# return {
			# 	"width": self.width,
			# 	"block_depth": self.block_depth,
			# }
			config = super(UpBlock, self).get_config()
			config.update({
				"width": self.width,
				"block_depth": self.block_depth,
			})
			return config


	def get_network(image_size, widths, block_depth):
		noisy_images = keras.Input(shape=(image_size, image_size, 3))
		noisy_variances = keras.Input(shape=(1, 1, 1))

		e = layers.Lambda(sinusoidal_embedding)(noisy_variances)
		e = layers.UpSampling2D(
			size=image_size, interpolation="nearest"
		)(e)

		x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
		x = layers.Concatenate()([x, e])

		skips = []
		for width in widths[:-1]:
			x, skips = DownBlock(width, block_depth)([x, skips])

		for _ in range(block_depth):
			x = ResidualBlock(widths[-1])(x)

		for width in reversed(widths[:-1]):
			x, skips = UpBlock(width, block_depth)([x, skips])

		x = layers.Conv2D(
			3, kernel_size=1, kernel_initializer="zeros"
		)(x)

		return keras.Model(
			inputs=[noisy_images, noisy_variances], outputs=x, 
			name="residual_unet"
		)


	# This showcases the power of the Functional API. Note how we built
	# a relatively complext U-Net with skip connectios, residual
	# blocks, multiple inputs, and sinusoidal embeddings in 80 lines of
	# code.


	# Diffusion Model
	# Diffusion schedule
	# Let's say that a diffusion process starts at time = 0 and ends at
	# time = 1. This variable will be called diffusion time, and can be
	# either discrete (common in diffusion models) or continuous
	# (common in score-based models). This example chooses the latter,
	# so that the number of sampling steps can be changed at inference
	# time.
	# We need to have a function that tells us at each point in the
	# diffusion process the noise levels and signal levels of the noisy
	# image corresponding to the actual diffusion time. This will be
	# called the diffusion schedule (see diffusion_schedule()).
	# This schedule outputs two quantities: the noise_rate and the
	# signal_rate (corresponding to sqrt(1 - alpha) an sqrt(alpha) in
	# the DDIM paper, respectively). We generate the noisy image by
	# weighting the random noise and the training image by their
	# corresponding rates and adding them together.
	# Since the (standard normal) random noises and the (normalized)
	# images both have zero mean and unit variance, the noise rate and
	# signal rate can be interpreted as the standard deviation of their
	# components in the noisy image, while the squares of their rates
	# can be interpreted as their variance (or their power in the
	# signal processing sense). The rates will always be set so that
	# their squared sum is 1, meaning that the noisy images will always
	# have unit variance, just like its unscaled components.
	# We will use a simplified, continuous version of the cosine
	# schedule (Section 3.2), that is quite commonly used in the
	# literature. This schedule is symmetric, slow towards the start
	# and end of the diffusion process, and it also has a nice
	# geometric interpretation, using the trigonometric properties of
	# the unit circle.
	# Training process
	# The training procedure (see train_step() and denoise()) of
	# denoising diffusion models is the following: we sample random
	# diffusion times uniformly, and mix the training images with
	# random gaussian noises at rates corresponding to the diffusion
	# times. Then, we train the model to separate the noisy image to
	# its two components.
	# Usually, the neural network is trained to predict the unscaled
	# noise component, from which the predicted image component can be
	# calculated using the signal and noise rates. Pixelwise mean
	# squared error should be used theoretically, however this example
	# recommends using mean absolute error instead (similarly to 
	# the implementation from lucidrains: https://github.com/
	# lucidrains/denoising-diffusion-pytorch/blob/main/denoising_
	# diffusion_pytorch/denoising_diffusion_pytorch.py#L371), which
	# produces better results on this dataset.
	# Sampling (reverse diffusion)
	# When sampling (see reverse_diffusion()), at each step we take the
	# previous estimate ofthe noisy image and separate it into image 
	# and noise using our network. Then we recombine these components
	# using the signal and noise rate of the following step.
	# Though a similar view is shown in Equation 12 of DDIMs, the above
	# explanation of the sampling equation may not be widely known.
	# This example only implements the deterministic sampling procedure
	# from DDIM, which corresponds to eta = 0 in the paper. One can
	# also use stochastic sampling (in which caes the model becomes a
	# Denoising Diffusion Probabilistic Model (DDPM)), where a part of
	# the predicted noise is replaced with the same or larger amount of
	# randome noise (see Equation 16 and below).
	# Stochastic sampling can be used without retraining the network
	# (since both models are trained the same way), and it can improve
	# sample quality, while on the other hand requiring more sampling
	# steps usually.
	class DiffusionModel(keras.Model):
		def __init__(self, image_size, widths, block_depth):
			super().__init__()

			self.normalizer = layers.Normalization()
			self.network = get_network(image_size, widths, block_depth)
			# self.ema_network = keras.models.clone_model(self.network)
			self.ema_network = get_network(
				image_size, widths, block_depth
			)
			self.ema_network.set_weights(self.network.get_weights())
			self.ema_network._name = "ema_residual_unet"


		def compile(self, **kwargs):
			super().compile(**kwargs)

			self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
			self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
			self.kid = KID(name="kid")


		@property
		def metrics(self):
			return [
				self.noise_loss_tracker, self.image_loss_tracker, 
				self.kid
			]


		def denormalize(self, images):
			# Convert the pixel values back to 0-1 range.
			images = self.normalizer.mean + images *\
				self.normalizer.variance ** 0.5
			return tf.clip_by_value(images, 0.0, 1.0)


		def diffusion_schedule(self, diffusion_times):
			# Diffusion times => angles.
			start_angle = tf.math.acos(max_signal_rate)
			end_angle = tf.math.acos(min_signal_rate)

			diffusion_angles = start_angle + diffusion_times *\
				(end_angle - start_angle)

			# Angles -> signal and noise rates.
			signal_rates = tf.math.cos(diffusion_angles)
			noise_rates = tf.math.sin(diffusion_angles)
			# Note that their squared sum is always sin^2(x) + cos^2(x)
			# = 1.

			return noise_rates, signal_rates


		def denoise(self, noisy_images, noise_rates, signal_rates, 
				training):
			# The exponential moving average weights are used at
			# evaluation.
			if training:
				network = self.network
			else:
				network = self.ema_network

			# Predict noise component and calculate the image component
			# using it.
			pred_noises = network(
				[noisy_images, noise_rates ** 2], training=training
			)
			pred_images = (noisy_images - noise_rates * pred_noises) /\
				signal_rates

			return pred_noises, pred_images


		def reverse_diffusion(self, initial_noise, diffusion_steps):
			# Reverse diffusion = sampling.
			num_images = initial_noise.shape[0]
			step_size = 1.0 / diffusion_steps

			# Important line:
			# At the first sampling step, the "noisy image" is pure
			# noise but its signal rate is assumed to be nonzero
			# (min_signal_rate).
			next_noisy_images = initial_noise
			for step in range(diffusion_steps):
				noisy_images = next_noisy_images

				# Separate the current noisy image to its components.
				diffusion_times = tf.ones((num_images, 1, 1, 1)) -\
					step * step_size
				noise_rates, signal_rates = self.denoise(
					noisy_images, noise_rates, signal_rates, 
					training=False
				)
				# Network used in eval mode.

				# Remix the predicted components using the next signal
				# and noise rates.
				next_diffusion_times = diffusion_steps - step_size
				next_noise_rates, next_signal_rates = self.diffusion_schedule(
					next_diffusion_times
				)
				next_noisy_images = (
					next_signal_rates * pred_images +next_noise_rates *
						pred_noises
				)
				# This new noisy image will be used in the next step.

			return pred_images


		def generate(self, num_images, diffusion_steps):
			# Noise -> images -> denormalized images.
			initial_noise = tf.random.normal(
				shape=(num_images, image_size, image_size, 3)
			)
			generated_images = self.reverse_diffusion(
				initial_noise, diffusion_steps
			)
			generated_images = self.denormalize(generated_images)
			return generated_images


		def train_step(self, images):
			# Normalize images to have standard deviation of 1, like
			# the noises.
			images = self.normalizer(images, training=True)
			noises = tf.random.normal(
				shape=(batch_size, image_size, image_size, 3)
			)

			# Sample uniform random diffusion times.
			diffusion_times = tf.random.uniform(
				shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
			)
			noise_rates, signal_rates = self.diffusion_schedule(
				diffusion_times
			)

			# Mix the images with noise accordingly.
			noisy_images = signal_rates * images + noise_rates * noises

			with tf.GradientTape() as tape:
				# Train the network to separate noisy images to their
				# components.
				pred_noises, pred_images = self.denoise(
					noisy_images, noise_rates, signal_rates, 
					training=True
				)

				noise_loss = self.loss(noises, pred_noises) # used for training
				image_loss = self.loss(noises, pred_noises) # only used as metric

			gradients = tape.gradient(
				noise_loss, self.network.trainable_weights
			)
			self.optimizer.apply_gradients(zip(
				gradients, self.network.trainable_weights
			))

			self.noise_loss_tracker.update_state(noise_loss)
			self.image_loss_tracker.update_state(image_loss)

			# Track the exponential moving averages of weights.
			for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
				ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

			# KID is not measured during the training phase for
			# computational efficiency.
			return {m.name: m.result() for m in self.metrics[:-1]}


		def test_step(self, images):
			# Normalize images to have standard deviation of 1, like
			# the noises.
			images = self.normalizer(images, training=True)
			noises = tf.random.normal(
				shape=(batch_size, image_size, image_size, 3)
			)

			# Sample uniform random diffusion times.
			diffusion_times = tf.random.uniform(
				shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
			)
			noise_rates, signal_rates = self.diffusion_schedule(
				diffusion_times
			)

			# Mix the images with noise accordingly.
			noisy_images = signal_rates * images + noise_rates * noises

			# Use the network to separate noisy images to their
			# components.
			pred_noises, pred_images = self.denoise(
				noisy_images, noise_rates, signal_rates, training=False
			)

			noise_loss = self.loss(noises, pred_noises)
			image_loss = self.loss(noises, pred_noises)

			self.noise_loss_tracker.update_state(noise_loss)
			self.image_loss_tracker.update_state(image_loss)

			# Measure KID between real and generated images. This is
			# computationally demanding, kid_diffusion_steps has to be
			# small.
			images = self.denormalize(images)
			generated_images = self.generate(
				num_images=batch_size,
				diffusion_steps=kid_diffusion_steps
			)
			self.kid.update_state(images, generated_images)

			return {m.name: n.result() for m in self.metrics}


		def plot_images(self, epoch=None, logs=None, num_rows=3, 
				num_cols=6):
			# Plot random generated images for visual evaluation of
			# generation quality.
			generated_images = self.generate(
				num_images=num_rows * num_cols,
				diffusion_steps=plot_diffusion_steps,
			)

			plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
			for row in range(num_rows):
				for col in range(num_cols):
					index = row * num_cols + col
					plt.subplot(num_rows, num_cols, index + 1)
					plt.imshow(generated_images[index])
					plt.axis("off")
			plt.tight_layout()
			# plt.show()
			plt.savefig(f"{epoch}_img_plot.png", "png")
			plt.close()


	# Training
	# Create and compile the model.
	model = DiffusionModel(image_size, widths, block_depth)

	# Anything below Tensorflow 2.9:
	# pip install tensorflow_addons
	# import tensorflow_addons as tfa
	# optimizer = tfa.optimizers.AdamW
	# model.compile(
	# 	optimizer=keras.optimizers.experimental.AdamW(
	# 		learning_rate=learning_rate, weight_decay=weight_decay
	# 	),
	# 	loss=keras.losses.mean_absolute_error,
	# )
	model.compile(
		optimizer=tfa.optimizers.AdamW(
			learning_rate=learning_rate, weight_decay=weight_decay
		),
		loss=keras.losses.mean_absolute_error,
	)
	# Pixelwise mean absolute error is used as loss.

	model.network.summary()

	# Save the best model based on the validation KID metric.
	checkpoint_path = "checkpoints/diffusion_model"
	checkpoint_callback = keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_path,
		save_weights_only=True,
		monitor="val_kid",
		mode="min",
		save_best_only=True,
	)

	# Calculate mean and variance of training dataset for
	# normalization.
	model.normalizer.adapt(train_dataset)

	# Run training and plot generated images periodically.
	model.fit(
		train_dataset,
		epochs=num_epochs,
		validation_data=val_dataset,
		callbacks=[
			keras.callbacks.LambdaCallback(
				on_epoch_end=model.plot_images
			),
			checkpoint_callback,
		],
	)

	# Inference
	# Load the best model and generate images.
	model.load_weights(checkpoint_path)
	model.plot_images()


	# Results
	# By running the training for at least 50 epochs (takes 2 hours on
	# a T4 GPU and 30 minutes on an A100 GPU), one can get high quality
	# image generations using this code example.
	# The original page presents samples and images under the following
	# settings:
	# -> The evolution of a batch of images over a 80 epoch training
	#	(color artifacts are due to GIF compression).
	# -> Images generated using between 1 and 20 sampling steps from the
	#	same initial noise.
	# -> Interpolation (spherical) between initial noise samples.
	# -> Deterministic sampling process (noisy images on top, predicted
	#	images on bottm, 40 steps).
	# -> Stochastic sampling process (noisy images on top, predicted 
	#	images on bottom, 80 steps).


	# Lessons Learned
	# During preparations for this code example, the author had run
	# numerous experiments using this repository: https://github.com/
	# beresandras/clear-diffusion-keras. This section lists the lessons
	# learned and the author's recommendations in their subjective
	# order of importance.
	# Algorithmic tips
	# -> min and max signal rates: The min signal rate was found to be
	#	an important hyperparameter. Setting it too low will make the
	#	generated iamge oversaturated, while setting it too high will
	#	make them undersaturated. They recommend tuning it carefully.
	#	Also, setting it to 0 will lead to a division by zero error.
	#	The max signal rate can be set to 1, but they found that 
	#	setting it lower slightly improves generation quality.
	# -> loss function: While large models tend to use mean squared
	#	error (MSE) loss, the author recommends using mean absolute
	#	error (MAE) on this dataset. In their experience, MSE loss
	#	generates more diverse samples (it also seems to hallucinate
	#	more Section 3), while MAE loss leads to smoother images. They
	#	recommend trying both.
	# -> weight decay: The author did occaisonally run into diverged
	#	trainings when scaling up the model, and found that weight 
	#	decay helps in avoiding instabilities at a low performance 
	#	cost. This is why they use AdamW instead of Adam in this 
	#	example.
	# -> exponential moving average weights: This helps to reduce the
	#	variance of the KID metric, and helps in averaging out 
	#	short-term changes during training.
	# -> image augmentations: Though they did not use image
	#	augmentations in this example, in their experience adding
	#	horizontal flips to the training increases generation
	#	performance, while random crops do not. Since we use a
	#	supervised denoising loss, overfitting can be an issue, so
	#	image augmentations might be important on small datasets. One
	#	should also be careful not to use leaky augmentations, which
	#	can be done following this method (end of Section 5) for
	#	inference.
	# -> data normalization: In the literature the pixel values of 
	#	images are usually converted to the -1 to 1 range. For 
	#	theoretical correctness, the author normalizated the images to
	#	have zero mean and unit variance instead, exactly like the
	#	random noises.
	# -> noise level input: The author chose to input the noise 
	#	variance to the network, as it is symmetrical under our 
	#	sampling schedule. One could also input the noise rate (similar
	#	performance), the signal rate (lower performance), or even the
	#	log-signal-to-noise ratio (Appendix B.1) (did not try, as its
	#	range is highly dependent on the min and max signal rates, and
	#	would require adjusting the min embedding frequency 
	#	accordingly).
	# -> gradient clipping: Using global gradient clipping with a value
	#	of 1 can help ith training stability for large models, but
	#	decreased performance significantly in their experience.
	# -> residual connection downscaling: For deper models (Appendix 
	#	B), scaling the residual connections with 1/sqrt(2) can be 
	#	helpful, but did not help in their case.
	# -> learning rate: For the author, Adam's optimizer's default
	#	learning rate of 1e-3 worked very well, but lower learning 
	#	rates are more common in the literature (Tables 11-13).
	# Architectureal tips
	# -> sinusoidal embedding: Using sinusoidal embeddings on the noise
	#	level input of the network is crucial for good performance. The
	#	author recommends setting the min embedding frequency to the
	#	reciprocal of the range of this input, and since we use the
	#	noise variance in this example, it can be left always at 1. The
	#	max embedding frequency controls the smallest change in the
	#	noise variance that the network will be sensitive to, and the
	#	embedding dimensions set the number of frequency components in
	#	the embedding. In their experience the performance is not too
	#	sensitive to these values.
	# -> skip connections: Using skip connections in the network
	#	architecture is absolutely critical, without them the model
	#	will fail to learn to denoise at a good performance.
	# -> residual connections: In their experience, residual 
	#	connections also significantly improve performance, but this 
	#	might be due to the fact that we only input the noise level
	#	embeddings to the first layer of the network instead of to all
	#	of them.
	# -> normalization: When scaling up the model, the author did
	#	occasionally encounter diverged trainings, using normalization
	#	layers helped to mitigate this issue. In the literature it is
	#	common to use GroupNormalization (with 8 groups for example) or
	#	LayerNormalization in the network, They however chose to use
	#	BatchNormalization, as it gave similar benefits in their
	#	experiments but was computationally higher.
	# -> activations: The choice of activation functions had a larger
	#	effect on generation quality that they expected. In their
	#	experiments using non-monotonic activation functions
	#	outperformed monotonic ones (such as ReLU), with Swish
	#	peforming the best (this is also what Imagen uses, page 41).
	# -> attention: As mentioned earlier, it is common in the
	#	literature to use attention layers at low resolutions for
	#	better global coherence. The author omitted them for 
	#	simplicity.
	# -> upsampling: Bilinear and nearest neighbor upsampling in the
	#	network performed similarly. The author did not try
	#	transposed convolutions however.
	# For a similar list about GANs check out this keras tutorial:
	# https://keras.io/examples/generative/gan_ada/#gan-tips-and-tricks


	# What to try next?
	# If you would like to dive in deeper to the topic, a recommend 
	# checking out this repository that I created in preparation for
	# this code example, which implements a wider range of features in
	# a similar style, such as:
	# -> stochastic sampling
	# -> second-order sampling based on the differential equation view
	#	of DDIMs (Equation 13)
	# -> more diffusion schedules
	# -> more network output types: predicting image or velocity 
	#	(Appendix D) instead of noise
	# -> more datasets


	# Related works
	# Score-based generative modeling (blogpost)
	# What are diffusion models? (blogpost)
	# Annotated diffusion model (blogpost)
	# CVPR 2022 tutorial on diffusion models (slides avaliable)
	# Elucidating the Design Space of Diffusion-Based Generative Models: attempts unifying diffusion methods under a common framework
	# High-level video overviews: 1, 2
	# Detailed technical videos: 1, 2
	# Score-based generative models: NCSN, NCSN+, NCSN++
	# Denoising diffusion models: DDPM, DDIM, DDPM+, DDPM++
	# Large diffusion models: GLIDE, DALL-E 2, Imagen


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()