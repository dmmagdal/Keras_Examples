# next_frame_prediction.py
# This program will train a model to predict the next frame of an
# artifically generated movie which contains moving squares.
# Source: https://keras.io/examples/vision/conv_lstm/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
	# Define the model.
	model = keras.models.Sequential([
			layers.Input(shape=(None, 40, 40, 1)),
			layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", 
								return_sequences=True),
			layers.BatchNormalization(),
			layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", 
								return_sequences=True),
			layers.BatchNormalization(),
			layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", 
								return_sequences=True),
			layers.BatchNormalization(),
			layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", 
								return_sequences=True),
			layers.BatchNormalization(),
			layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", 
							padding="same")
		])
	model.compile(loss="binary_crossentropy", optimizer="adadelta")
	print(model.summary())

	# Generate artificial data (movies with 3 to 7 moving squares
	# inside). The squares are of shape 1x1 or 2x2 pixels, and move
	# linearly over time. For convenience, first create movies with
	# bigger width and height (80x80) and at the end select a 40x40
	# window.
	noisy_movies, shifted_movies = generate_movies(n_samples=1200)

	# Train the model.
	epochs = 1 # In practice, would need hundreds of epochs.
	model.fit(noisy_movies[:1000], shifted_movies[:1000], batch_size=10, 
				epochs=epochs, verbose=2, validation_split=0.1)

	# Text the model on one movie.
	movie_index = 1004
	test_movie = noisy_movies[movie_index]

	# Start from first 7 frames.
	track = test_movie[:7, ::, ::, ::]

	# Predict 16 frames.
	for j in range(16):
		new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
		new = new_pos[::, -1, ::, ::, ::]
		track = np.concatenate((track, new), axis=0)
	
	# Exit the program
	exit(0)


def generate_movies(n_samples=1200, n_frames=15):
	row = 80
	col = 80
	noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
	shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), 
								dtype=np.float)

	for i in range(n_samples):
		# Add 3 to 7 moving squares.
		n = np.random.randint(3, 8)

		for j in range(n):
			# Initial position.
			xstart = np.random.randint(20, 60)
			ystart = np.random.randint(20, 60)

			# Direction of motion.
			directionx = np.random.randint(0, 3) - 1
			directiony = np.random.randint(0, 3) - 1

			# Size of the square.
			w = np.random.randint(2, 4)

			for t in range(n_frames):
				x_shift = xstart + directionx * t
				y_shift = ystart + directiony * t
				noisy_movies[i, t, x_shift - w:x_shift + w,
							y_shift - w:y_shift + w, 0] += 1

				# Make it more robust by adding noise. The idea is that
				# if during inference, the value of the pixel is not
				# exactly one, we need to train the model to be robust
				# and still consider it as a pixel belonging to a
				# square.
				if np.random.randint(0, 2):
					noise_f = (-1) ** np.random.randint(0, 2)
					noisy_movies[i, t, x_shift - w:x_shift + w,
								y_shift - w: y_shift - w, 0] += 1

		# Cut to a 40x40 window.
		noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
		shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
		noisy_movies[noisy_movies >= 1] = 1
		shifted_movies[shifted_movies >= 1] = 1
		return noisy_movies, shifted_movies


if __name__ == '__main__':
	main()