# image_classification_efficientnet.py
# Use EfficientNet with weights pre-trained on imagenet for Stanford
# Dogs classification.
# Introduction: What is EfficientNet?
# EfficientNet, first introduced in Tan and Le, 2019 is among the most
# efficient models (i.e. requiring least FLOPS for inference) that
# reaches State-of-the-Art accuracy on both imagenet and common image
# classification transfer learning tasks.
# The smallest base model is similar to MnasNet, which reached
# near-SOTA with a significantly smaller model. By introducing a
# heuristic way to scale the model, EfficientNet provides a family of
# models (B0 to B7) that represents a good combination of efficiency
# and accuracy on a variety of scales. Such a scaling heuristic
# (compound-scaling), details see Tan and Le, 2019) allows the
# efficiency-oriented base model (B0) to surpass models at every scale,
# while avoiding extensive grid-search of hyperparameters.
# A summary of the latest updates on the model is available here, where
# various augmentation schemes and semi-supervised learning approaches
# are applied to further improve the imagenet performance of the
# models. These extensions of the model can be used by updating weights
# without changing model architecture.
# B0 to B7 varients of EfficientNet.
# (This section provides some details on "compound scaling", and can be
# skipped if only interested in using models). Based on the original
# paper people may have the impression that Efficientnet is a
# continuous family of models created by arbitrarily choosing scaling
# factor is as Eq(3) of the paper. However, choice of resolution, depth
# and width are also restricted by many factors:
# -> Resolution: Resolutions are not divisible by 8, 16, etc. cause
#	zero-padding new boundaries of some layers which wastes 
#	computational resources. This especially applies to smaller 
#	variants of the model, hence the input resolution for B0 and B1 are
#	chosen as 224 and 240.
# -> Depth and width: The building blocks of EfficientNet demands 
#	channel size to be multiples of 8.
# -> Resource limit: Memory limitation my bottleneck resolution when 
#	depth and width can still increase. In such a situation, increasing
#	depth and/or width but keep resolution can still improve 
#	performance.
# As a result, the depth, width, and resolution of each variant of the
# EfficientNet models are hand-picked and proven to produce good
# results, though they may be significantly off from the compound
# scaling formula. Therefore, the keras implementation (detailed below)
# only provide these 8 models, B0 to B7, instead of allowing arbitrary
# choice of width/ depth/ resolution parameters.
# Keras implementation of EfficientNet.
# An implementation of EfficientNet B0 and B7 has been shipped with
# tf.keras since TF 2.3. To use EfficientNetB0 for classifying 1000
# classes of images from imagenet, run:
# from tensorflow.keras.applications import EfficientNetB0
# model = EfficientNetB0(weights="imagenet")
# This model takes input images of shape (224, 224, 3), and the input
# data should range [0, 255]. Normalization is included as part of the
# model.
# Because training EfficientNet on ImageNet takes a tremendous amount
# of resources and several techniques that are not a part of the model
# architecture itself. Hence the Keras implementation by defaults loads
# pre-trained weights obtained via training with AutoAugment. For B0 to
# B7 models, the input shapes are different. Here is a list of input
# shape expected for each model:
#    Base model    Resolution
#    B0            224
#    B1            240
#    B2            260
#    B3            300
#    B4            380
#    B5            456
#    B6            528
#    B7            600
# When the model is intended for transfer learning, the Keras
# implementation provides a option to remove the top layers:
# model = EfficientNetB0(include_top=False, weights="imagenet")
# This option excludes the final Dense layer that turns 1280
# features on the penultimate layer into prediction of the 1000
# ImageNet classes. Replacing the top layer with custom layers allows
# using EfficientNet as a feature extractor in a transfer learning
# workflow.
# Another argument in the model constructor worth noticing is
# drop_connect_rate which controls the dropout rate responsible for
# stochastic depth. This parameter serves as a toggle for extra
# regularization in finetuning, but does not affect loaded weights. For
# example, when stronger regularization is desired, try:
# model = EfficientNetB0(weights="imagenet", drop_connect_rate=0.4)
# The default value is 0.2.
# Example: EfficientNetB0 for Stanford Dogs.
# EfficientNet is capable of a wide range of image classification
# tasks. This makes it a good model for transfer learning. As an
# end-to-en example, we will show using pre-trained EfficientNetB0 on
# Stanford Dogs dataset.
# # IMG_SIZE is determined by EfficientNet model choice
# IMG_SIZE = 224
# This example uses Tensorflow 2.3 or above. To use TPU, the TPU
# runtime must match current running Tensorflow version. If there is a
# mismatch, try:
# from cloud_tpu_client import Client
# c = Client()
# c.configure_tpu_version(tf.__version__, restart_type="always")
# import tensorflow as tf
# try:
#	tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
#	print("Running on TPU", tpu.cluster_spec().as_dict()["worker"])
#	tf.config.experimental_connect_to_cluster(tpu)
#	tf.tpu.experimental.initialize_tpu_system(tpu)
#	strategy = tf.distribute.experimental.TPUStrategy(tpu)
# except ValueError:
#	print("Not connected to a TPU runtime. Using CPU/GPU strategy")
#	strategy = tf.distribute.MirroredStrategy()
# Source: https://keras.io/examples/vision/image_classification_
# efficientnet_fine_tuning/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import matplotlib.pyploy as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0


def main():
	# Loading data.
	# Load data from tensorflow_dataset (hereafter TFDS). Stanford Dogs
	# dataset is provided in TFDS as stanford_dogs. It features 20,580
	# images that belong to 120 classes of dog breeds (12,000 for
	# training and 8,580 for testing).
	# By simply changing dataset_name below, you may also try this
	# notebook for other datasets in TFDS such as cifar10, cifar100,
	# food101, etc. When the images are much smaller than the size of
	# Efficientnet input, we can simply upsample the input images. It
	# has been shown in Tan and Le, 2019 that transfer learning result
	# is better for increased resolution even if input images remain
	# small.
	# For TPU: if using TFDS datasets, a GCS bucket location is
	# required to save the datasets. For example:
	# tfds.load(dataset_name, data_dir="gs://example-bucket/datapath")
	# Also, both the current environment and the TPU service account
	# have proper access to the bucket. Alternatively, for small
	# datasets you may try loading data into the memory and use
	# tf.data.Dataset.from_tensor_slices().
	batch_size = 64

	dataset_name = "stanford_dogs"
	(ds_train, ds_test), ds_info = tfds.load(
		dataset_name, split=["train", "test"], with_info=True, as_supervised=True
	)
	num_classes = ds_info.features["label"].num_classes

	# When the dataset include images with various size, need to resize
	# them into a shared size. The Stanford Dogs dataset includes only
	# images at least 200x200 pixels in size. Here, resize the images
	# to the input size needed for EfficientNet.
	size = (IMG_SIZE, IMG_SIZE)
	ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
	ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

	# Visualize the data.
	# The following code shows the first 9 images with their labels.
	#'''
	def format_label(label):
		string_label = label_info.int2str(label)
		return string_label.split("-")[1]

	label_info = ds_info.features["labels"]
	for i, (image, label) in enumerate(ds_train.take(9)):
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(image.numpy().astype("uint8"))
		plt.title("{}".format(format_label(label)))
		plt.axis("off")
	#'''

	# Data augmentation.
	# Use preprocessing layers APIs for image augmentation.
	img_augmentation = Sequential(
		[
			preprocessing.RandomRotation(factor-0.15),
			preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
			preprocessing.RandomFlip(),
			preprocessing.RandomContrast(factor=0.1),
		],
		name="img_augmentation",
	)

	# This sequential model object can be used both as part of the
	# model built later, and as a function to preprocess data before
	# feeding into the model. Using them as a function makes it easy to
	# visualize the augmented images. Here, plot 9 examples of
	# augmentation result of a given figure.
	#'''
	for image, label in ds_train.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			aug_img = img_augmentation(tf.expand_dims(image, axis=0))
			plt.imshow(image[0].numpy().astype("uint8"))
			plt.title("{}".format(format_label(label)))
			plt.axis("off")
	#'''

	# Prepare inputs.
	# Once verified the input data and augmentation are working
	# correctly, prepared dataset for training. The input data is
	# resized to uniform IMG_SIZE. The labels are put into a one-hot
	# (a.k.a categorical) encoding. The dataset is batched.
	# Note: prefetch and AUTOTUNE may in some situation improve
	# performance, but depends on environment and the specific dataset
	# used. See this guide for more information on data pipeline
	# performance.
	# One-hot / categorical encoding.
	def input_preprocess(image, label):
		label = tf.one_hot(label, num_classes)
		return image, label

	ds_train = ds_train.map(
		input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
	)
	ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
	ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

	ds_test = ds_test.map(input_preprocess)
	ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

	# Train a model from scratch.
	# Build an EfficientNetB0 with 120 output classes, that is
	# initialized from scratch.
	# Note: the accuracy will increase very slowly and may overfit.
	with strategy.scope():
		inputs = la


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()