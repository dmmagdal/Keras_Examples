# pixelCNN.py
# PixelCNN implemented in Keras.
# PixelCNN is a generative model proposed in 2016 by van den Oord et
# al. (reference: Conditional Image Generation with PixelCNN Decoders).
# It is designed to generate images (or other data types) iteratively
# from the probability distribution of later elements. In the following
# example, images are generated in this fashion, pixel-by-pixel, via a
# masked convolution kernel that only looks at data from previously
# generated pixels (origin at the top left) to generate later pixels.
# During inference, the output of the network is used as a probability
# distribution from which new pixel values are sampled to generate a
# new image (here, with MNIST, the pixel values range from white(0) to
# black (255)).
# Source: https://keras.io/examples/generative/pixelcnn/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from IPython.display import Image, display


def main():
	# Get the data.
	# Model/data parameters.
	num_classes = 10
	input_shape = (28, 28, 1)
	n_residual_blocks = 5

	# The data, split between train and test sets.
	(x, _), (y, _) = keras.datasets.mnist.load_data()

	# Concatenate all of the images together.
	data = np.concatenate((x, y), axis=0)

	# Round all pixel values less than 33% of the max 256 value to 0.
	# Anything above this value gets rounded up to 1 so that all values
	# are eighter 0 or 1.
	data = np.where(data < (0.33 * 256), 0, 1)
	data = data.astype(np.float32)

	# Build the model based on the original paper.
	inputs = keras.Input(shape=input_shape)
	x = PixelConvLayer(
		mask_type="A", filters=128, kernel_size=7, activation="relu", 
		padding="same"
	)(inputs)

	for _ in range(n_residual_blocks):
		x = ResidualBlock(filters=128)(x)

	for _ in range(2):
		x = PixelConvLayer(
			mask_type="B", filters=128, kernel_size=1, strides=1,
			activation="relu", padding="same"
		)(x)

	out = layers.Conv2D(
		filters=1, kernel_size=1, strides=1, activation="sigmoid", padding="valid"
	)(x)

	pixel_cnn = keras.Model(inputs, out)
	adam = keras.optimizers.Adam(learning_rate=0.0005)
	pixel_cnn.compile(optimizer=adam, loss="binary_crossentropy")

	pixel_cnn.summary()
	pixel_cnn.fit(
		x=data, y=data, batch_size=128, epochs=50, validation_split=0.1, verbose=2
	)

	# Demonstration.
	# The PixelCNN cannot generate the full image at once. Instead, it
	# must generate each pixel in order, append the last generated
	# pixel to the current image, and feed the image back into the
	# model to repeat the process.
	# Create an empty array of pixels.
	batch = 4
	pixels = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
	batch, rows, cols, channels = pixels.shape

	# Iterate over the pixels because generation has to be done
	# sequentially pixel by pixel.
	for row in tqdm(range(rows)):
		for col in tqdm(range(cols)):
			for channel in range(channels):
				# Feed the whole array and retrieving the pixel value
				# probabilities for the next pixel.
				probs = pixel_cnn.predict(pixels)[:, row, col, channel]

				# Use the probabilities to pick pixel values and append
				# the values to the image frame.
				pixels[:, row, col, channel] = tf.math.ceil(
					probs - tf.random.uniform(probs.shape)
				)

	# Iterate over the generated images and plot them with matplotlib.
	for i, pic in enumerate(pixels):
		keras.preprocessing.image.save_img(
			"generated_image_{}.png".format(i), deprocess_image(np.squeeze(pic, -1))
		)

	display("generated_image_0.png")
	display("generated_image_1.png")
	display("generated_image_2.png")
	display("generated_image_3.png")

	# Exit the program.
	exit(0)


# Create two classes for the requisite layers for the model.
# The first layer is the PixelCNN layer. This layer simply builds on
# the 2D convolutional layer, but includes masking.
class PixelConvLayer(layers.Layer):
	def __init__(self, mask_type, **kwargs):
		super(PixelConvLayer, self).__init__()
		self.mask_type = mask_type
		self.conv = layers.Conv2D(**kwargs)


	def build(self, input_shape):
		# Build the conv2d layer to initialize kernel variables.
		self.conv.build(input_shape)

		# Use the initialized kernel to create the mask.
		kernel_shape = self.conv.kernel.get_shape()
		self.mask = np.zeros(shape=kernel_shape)
		self.mask[: kernel_shape[0] // 2, ...] = 1.0
		self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
		if self.mask_type =="B":
			self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0


	def call(self, inputs):
		self.conv.kernel.assign(self.conv.kernel * self.mask)
		return self.conv(inputs)


# Next, build the residual block layer. This is just a normal residual
# block, but based on the PixelConvLayer.
class ResidualBlock(layers.Layer):
	def __init__(self, filters, **kwargs):
		super(ResidualBlock, self).__init__()
		self.conv1 = layers.Conv2D(
			filters=filters, kernel_size=1, activation="relu"
		)
		self.pixel_conv = PixelConvLayer(
			mask_type="B",
			filters=filters // 2,
			kernel_size=3,
			activation="relu",
			padding="same"
		)
		self.conv2 = layers.Conv2D(
			filters=filters, kernel_size=1, activation="relu"
		)


	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.pixel_conv(x)
		x = self.conv2(x)
		return layers.add([inputs, x])


def deprocess_image(x):
	# Stack the single channeled black and white image to RGB values.
	x = np.stack((x, x, x), 2)

	# Undo preprocessing.
	x *= 255.0

	# Convert to uint8 and clip to the valid range [0, 255].
	x = np.clip(x, 0, 255).astype("uint8")
	return x


if __name__ == '__main__':
	main()