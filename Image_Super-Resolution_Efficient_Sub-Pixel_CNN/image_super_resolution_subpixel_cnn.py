# image_super_resolution_subpixel_cnn.py
# Implements Super-Resolution using Efficient sub-pixel model on
# BSDS500.
# ESPCN (Efficient Sub-Pixel CNN), proposed by Shi, 2016 is a model
# that reconstructs a high-resolution version of an image given a low-
# resolution version. It leverages efficient "sub-pixel convolution"
# layers, which learns an array of image upscaling filters. This
# example will implement the model from the paper and train it on a
# small dataset, BSDS500.
# Source: https://keras.io/examples/vision/super_resolution_sub_pixel/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from IPython.display import display
#'''
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#'''
import PIL


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	# Load dataset.
	dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
	data_dir = keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)
	root_dir = os.path.join(data_dir, "BSDS500/data")

	# Create the training and validation datasets via
	# image_dataset_from_directory.
	crop_size = 300
	upscale_factor = 3
	input_size = crop_size // upscale_factor
	batch_size = 8

	train_ds = image_dataset_from_directory(
		root_dir,
		batch_size=batch_size,
		image_size=(crop_size, crop_size),
		validation_split=0.2,
		subset="training",
		seed=1337,
		label_mode=None,
	)

	valid_ds = image_dataset_from_directory(
		root_dir,
		batch_size=batch_size,
		image_size=(crop_size, crop_size),
		validation_split=0.2,
		subset="validation",
		seed=1337,
		label_mode=None,
	)

	# Rescale the images to take values in the range [0, 1].
	def scaling(input_image):
		input_image = input_image / 255.0
		return input_image

	# Scale from (0, 255) to (0, 1).
	train_ds = train_ds.map(scaling)
	valid_ds = valid_ds.map(scaling)

	# Visualze a few sample images.
	for batch in train_ds.take(1):
		for img in batch:
			display(array_to_img(img))

	# Prepare a dataset of test image paths that will be used for
	# evaluation at the end of the example.
	dataset = os.path.join(root_dir, "images")
	test_path = os.path.join(dataset, "test")

	test_img_paths = sorted(
		[
			os.path.join(test_path, fname)
			for fname in os.listdir(test_path)
			if fname.endswith(".jpg")
		]
	)

	# Crop and resize images.
	# Process the image data by converting the images from RGB color
	# space to YUV colour space. For the input data (low resolution
	# images), crop the image, retrieve the y channel (luminance), and
	# resixe it with the area method (use BICUBIC if using PIL). Only
	# consider the luminance channel in the YUV color space because
	# humans are more sensitive to luminance change. For the target
	# data (high-resolution images), crop the image and retrieve the y
	# channel.
	train_ds = train_ds.map(
		lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
	)
	train_ds = train_ds.prefetch(buffer_size=32)

	valid_ds = valid_ds.map(
		lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
	)
	valid_ds = valid_ds.prefetch(buffer_size=32)

	# View some of the input and target data.
	for batch in train_ds.take(1):
		for img in batch[0]:
			display(array_to_img(img))
		for img in batch[1]:
			display(array_to_img(img))

	# Define ModelCheckpoint and EarlyStopping callbacks.
	early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

	checkpoint_filepath = "./tmp/checkpoint"
	#os.mkdir("tmp")
	#os.mkdir(checkpoint_filepath)

	model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
		save_weights_only=True,
		monitor="loss",
		mode="min",
		save_best_only=True,
	)

	model = get_model(upscale_factor=upscale_factor, channels=1)
	model.summary()


	callbacks = [
		ESPCNCallback(test_img_paths, upscale_factor),
		early_stopping_callback, model_checkpoint_callback
	]
	loss_fn = keras.losses.MeanSquaredError()
	optimizer = keras.optimizers.Adam(learning_rate=0.001)

	# Train the model.
	epochs=100

	model.compile(optimizer=optimizer, loss=loss_fn)

	model.fit(
		train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
	)

	# The model weights (that are considered the best) are loaded into
	# the model.
	model.load_weights(checkpoint_filepath)

	# Run model prediction and plot the results.
	# Compute the reconstructed version of a few images and save the
	# results.
	total_bicubic_psnr = 0.0
	total_best_psnr = 0.0

	for index, test_img_path in enumerate(test_img_paths[50:60]):
		img = load_img(test_img_path)
		lowres_input = get_lowres_image(img, upscale_factor)
		w = lowres_input.size[0] * upscale_factor
		h = lowres_input.size[1] * upscale_factor
		highres_img = img.resize((w, h))
		prediction = upscale_image(model, lowres_input)
		lowres_img = lowres_input.resize((w, h))
		lowres_img_arr = img_to_array(lowres_img)
		highres_img_arr = img_to_array(highres_img)
		predict_img_arr = img_to_array(prediction)
		bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
		test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

		total_bicubic_psnr += bicubic_psnr
		total_best_psnr += test_psnr

		print(
			"PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr
		)
		print("PSNR of predict and high resolution is %.4f" % test_psnr)
		#'''
		plot_results(lowres_img, index, "lowres")
		plot_results(highres_img, index, "highres")
		plot_results(prediction, index, "predict")
		#'''

	print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
	print("Avg. PSNR of reconstructions is %.4f" % (total_best_psnr / 10))

	# Exit the program.
	exit(0)


# Use TF Ops to process.
def process_input(input, input_size, upscale_factor):
	input = tf.image.rgb_to_yuv(input)
	last_dimension_axis = len(input.shape) - 1
	y, u, v = tf.split(input, 3, axis=last_dimension_axis)
	return tf.image.resize(y, [input_size, input_size], method="area")


def process_target(input):
	input = tf.image.rgb_to_yuv(input)
	last_dimension_axis = len(input.shape) - 1
	y, u, v = tf.split(input, 3, axis=last_dimension_axis)
	return y


# Build a model.
# Compared to the paper, add one more layer and use the relu activation
# function instead of tanh. It achieves better performance even though
# the model is trained for fewer epochs.
def get_model(upscale_factor=3, channels=1):
	conv_args = {
		"activation": "relu",
		"kernel_initializer": "Orthogonal",
		"padding": "same",
	}
	inputs = keras.Input(shape=(None, None, channels))
	x = layers.Conv2D(64, 5, **conv_args)(inputs)
	x = layers.Conv2D(64, 3, **conv_args)(x)
	x = layers.Conv2D(32, 3, **conv_args)(x)
	x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
	outputs = tf.nn.depth_to_space(x, upscale_factor)
	return keras.Model(inputs, outputs)


# Define utility functions.
# Need to define several utility functions to monitor results:
# -> plot_results to plot and save an image.
# -> got_lowres_image to convert an image to its low-resolution
#	version.
# -> upscale_image to turn a low-resolution image to a high-resolution
#	version reconstructed by the model. In this function, use the y
#	channel from the YUV color space as input to the model and then
#	combine the output with the other channels to obtain an RGB image.
#'''
def plot_results(img, prefix, title):
	# Plot the result with zoom-in area.
	img_array = img_to_array(img)
	img_array = img_array.astype("float32") / 255.0

	# Create a new figure with a default 111 subplot.
	fig, ax = plt.subplots()
	im = ax.imshow(img_array[::-1], origin="lower")

	plt.title(title)
	# zoom-factor: 2.0, location: upper-left
	axins = zoomed_inset_axes(ax, 2, loc=2)
	axins.imshow(img_array[::-1], origin="lower")

	# Specify the limits.
	x1, x2, y1, y2 = 200, 300, 100, 200

	# Apply the x-limits.
	axins.set_xlim(x1, x2)

	# Apply the y-limits.
	axins.set_ylim(y1, y2)

	plt.yticks(visible=False)
	plt.xticks(visible=False)

	# Make the line.
	mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
	plt.savefig(str(prefix) + "-" + title + ".png")
	plt.show()
#'''


def get_lowres_image(img, upscale_factor):
	# Return low-resolution image to use as model input.
	return img.resize(
		(img.size[0] // upscale_factor, img.size[1] // upscale_factor),
		PIL.Image.BICUBIC,
	)


def upscale_image(model, img):
	# Predict the result based on input image and restore the image as
	# RGB.
	ycbcr = img.convert("YCbCr")
	y, cb, cr = ycbcr.split()
	y = img_to_array(y)
	y = y.astype("float32") / 255.0

	input = np.expand_dims(y, axis=0)
	out = model.predict(input)

	out_img_y = out[0]
	out_img_y *= 255.0

	# Restore the image in RGB color space.
	out_img_y = out_img_y.clip(0, 255)
	out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
	out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
	out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
	out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
	out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
		"RGB"
	)
	return out_img


# Define callbacks to monitor training.
# The ESPCNCallback object will compute and display the PSNR metric.
# This is the main metric used to evaluate super-resolution performance.
class ESPCNCallback(keras.callbacks.Callback):
	def __init__(self, test_img_paths, upscale_factor):
		super(ESPCNCallback, self).__init__()
		self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)


	# Store PSNR value in each epoch.
	def on_epoch_begin(self, epoch, logs=None):
		self.psnr = []


	def on_epoch_end(self, epoch, logs=None):
		print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
		if epoch % 20 == 0:
			prediction = upscale_image(self.model, self.test_img)
			#'''
			plot_results(prediction, "epoch-" + str(epoch), "prediction")
			#'''


	def on_test_batch_end(self, batch, logs=None):
		self.psnr.append(10 * math.log10(1 / logs["loss"]))


if __name__ == '__main__':
	main()