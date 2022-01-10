# lego_minifig_gan.py
# Simple keras GAN to create LEGO minifigures.
# Source: https://www.kaggle.com/christianlillelund/simple-keras-gan-
# to-create-lego-figures
# Data Source: https://www.kaggle.com/ihelon/lego-minifigures-
# classification
# Alt Project Source: https://medium.com/@thundo/generate-lego-
# minifigures-with-deep-learning-f14cb8d678d8
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
	# Hyperparameters and directory variables.
	INPUT_DIR = "./lego-minifigures-classification"
	OUTPUT_DIR = "./working"
	N_CHANNELS = 3
	LATENT_DIM = 100
	IMAGE_HEIGHT = 128
	IMAGE_WIDTH = 128
	BATCH_SIZE = 32
	NUM_EPOCHS = 1000
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	# Load the dataset.
	def load_lego_data():
		data_generator = keras.preprocessing.image.ImageDataGenerator(
			preprocessing_function=keras.applications.xception.preprocess_input,
			data_format="channels_last",
			rotation_range=0,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=False,
			vertical_flip=False,
		)

		flow_from_directory_params = {
			"target_size": (IMAGE_HEIGHT, IMAGE_WIDTH),
			"color_mode": "grayscale" if N_CHANNELS == 1 else "rgb",
			"class_mode": None,
			"batch_size": BATCH_SIZE
		}

		image_generator = data_generator.flow_from_directory(
			directory=INPUT_DIR,
			**flow_from_directory_params
		)

		return image_generator

	image_generator = load_lego_data()

	# Retrieve a batch of the data from the image generator.
	def get_image_batch(image_generator):
		img_batch = image_generator.next()

		if len(img_batch) != BATCH_SIZE:
			img_batch = image_generator.next()

		assert img_batch.shape == (
			BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS
		), img_batch.shape
		return img_batch

	# Generator.
	def make_generator():
		generator = keras.models.Sequential()

		# Foundation for 4x4 image.
		generator.add(layers.Dense(256 * 4 * 4, input_dim=LATENT_DIM))
		generator.add(layers.LeakyReLU(alpha=0.2))
		generator.add(layers.Reshape((4, 4, 256)))

		# Upsample to 8x8.
		generator.add(layers.Conv2DTranspose(
			128, (4, 4), strides=(2, 2), padding="same", use_bias=False
		))
		generator.add(layers.LeakyReLU(alpha=0.2))

		# Upsample to 16x16.
		generator.add(layers.Conv2DTranspose(
			128, (4, 4), strides=(2, 2), padding="same", use_bias=False
		))
		generator.add(layers.LeakyReLU(alpha=0.2))

		# Upsample to 32x32.
		generator.add(layers.Conv2DTranspose(
			128, (4, 4), strides=(2, 2), padding="same", use_bias=False
		))
		generator.add(layers.LeakyReLU(alpha=0.2))

		# Upsample to 64x64.
		generator.add(layers.Conv2DTranspose(
			128, (4, 4), strides=(2, 2), padding="same", use_bias=False
		))
		generator.add(layers.LeakyReLU(alpha=0.2))

		# Upsample to 128x128.
		generator.add(layers.Conv2DTranspose(
			128, (4, 4), strides=(2, 2), padding="same", use_bias=False
		))
		generator.add(layers.LeakyReLU(alpha=0.2))

		# Output 128x128.
		generator.add(layers.Conv2D(
			N_CHANNELS, (3, 3), activation="tanh", padding="same",
			use_bias=False
		))
		return generator

	# Discriminator.
	def make_discriminator():
		discriminator = keras.models.Sequential()

		# Normal.
		discriminator.add(layers.Conv2D(
			64, (3, 3), padding="same", 
			input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)
		))
		discriminator.add(layers.LeakyReLU(alpha=0.2))

		# Downsample to 64x64.
		discriminator.add(layers.Conv2D(
			64, (3, 3), strides=(2, 2), padding="same", 
		))
		discriminator.add(layers.LeakyReLU(alpha=0.2))

		# Downsample to 32x32.
		discriminator.add(layers.Conv2D(
			64, (3, 3), strides=(2, 2), padding="same",
		))
		discriminator.add(layers.LeakyReLU(alpha=0.2))

		# Downsample to 16x16.
		discriminator.add(layers.Conv2D(
			64, (3, 3), strides=(2, 2), padding="same", 
		))
		discriminator.add(layers.LeakyReLU(alpha=0.2))

		# Downsample to 8x8.
		discriminator.add(layers.Conv2D(
			64, (3, 3), strides=(2, 2), padding="same",
		))
		discriminator.add(layers.LeakyReLU(alpha=0.2))

		# Classifier.
		discriminator.add(layers.Dropout(0.3))
		discriminator.add(layers.Flatten())
		discriminator.add(layers.Dense(1, activation="sigmoid"))

		# Compile the discriminator.
		optimizer = keras.optimizers.RMSprop(
			learning_rate=0.0004, clipvalue=1.0, decay=1e-8
		)
		discriminator.compile(
			optimizer=optimizer, loss="binary_crossentropy",
			metrics=["accuracy"]
		)
		return discriminator

	# GAN training loop.
	def train_gan(gan, generator, discriminator, image_generator):
		# Run training loop for num_epochs.
		for epoch in range(NUM_EPOCHS):
			real_images = get_image_batch(image_generator)
			real_labels = np.ones((BATCH_SIZE, 1))
			d_loss1, _ = discriminator.train_on_batch(
				real_images, real_labels
			)

			random_latent_vectors = np.random.normal(
				size=(BATCH_SIZE, LATENT_DIM)
			)
			fake_data = generator.predict(random_latent_vectors)
			fake_labels = np.zeros((BATCH_SIZE, 1))
			d_loss2, _ = discriminator.train_on_batch(
				fake_data, fake_labels
			)

			random_latent_vectors = np.random.normal(
				size=(BATCH_SIZE, LATENT_DIM)
			)
			misleading_labels = np.ones((BATCH_SIZE, 1))
			g_loss = gan.train_on_batch(
				random_latent_vectors, misleading_labels
			)

			print("Epoch %d, d1=%.3f, d2=%.3f g=%.3f" % 
				(epoch + 1, d_loss1, d_loss2, g_loss)
			)

			# Save GAN weights, a fake and real image every 50 epochs.
			if epoch % 100 == 0:
				gan.save_weights(os.path.join(OUTPUT_DIR, "gan.h5"))
				print(f"Saving batch of generated images at adverserial step: {epoch}")
				random_latent_vectors = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
				fake_images = generator.predict(random_latent_vectors)
				real_images = get_image_batch(image_generator)
				img = keras.preprocessing.image.array_to_img(fake_images[0])
				img.save(os.path.join(OUTPUT_DIR, "generated_lego_epoch_" + str(epoch) + ".png"))
				img = keras.preprocessing.image.array_to_img(real_images[0])
				img.save(os.path.join(OUTPUT_DIR, "real_lego_epoch_" + str(epoch) + ".png"))

	# Make the GAN.
	generator = make_generator()
	discriminator = make_discriminator()
	discriminator.trainable = False
	gan = keras.models.Sequential([generator, discriminator])

	# Compile the GAN.
	optimizer = keras.optimizers.RMSprop(
		learning_rate=0.0008, clipvalue=1.0, decay=1e-8
	)
	gan.compile(optimizer=optimizer, loss="binary_crossentropy")

	# Train the GAN.
	train_gan(gan, generator, discriminator, image_generator)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()