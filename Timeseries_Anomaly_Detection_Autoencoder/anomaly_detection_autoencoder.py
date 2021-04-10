# anomaly_detection_autoencoder.py
# Use a reconstruction convolutional autoencoder model to detect
# anomalies in timeseries data.
# Source: https://keras.io/examples/timeseries/timeseries_anomaly_detection/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
#from matplotlib import pyplot as plt


def main():
	# Load the data. Use the Numenta Anomaly Benchmark (NAB) dataset.
	# It provides artificial timeseries data containing labeled
	# anomalous periods of behavior. Data rae ordered, timestamped,
	# single-valued metrics. Use the art_daily_small_noise.csv file for
	# training and the art_daily_jumpsup.csv file for testing. The
	# simplicity of this dataset allows us to demonstrate anomaly
	# detection effectively.
	master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

	df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
	df_small_noise_url = master_url_root + df_small_noise_url_suffix
	df_small_noise = pd.read_csv(
		df_small_noise_url, parse_dates=True, index_col="timestamp"
	)

	df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
	df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
	df_daily_jumpsup = pd.read_csv(
		df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
	)

	# Quick look at the data.
	print(df_small_noise.head())
	print(df_daily_jumpsup.head())

	# Visualize the data. Timeseries data without anomalies will be
	# used for training.
	'''
	fig, ax = plt.subplots()
	df_small_noise.plot(legend=False, ax=ax)
	plt.show()
	'''

	# Timeseries data with anomalies will be used for testing and see
	# if the sudden jump up in the data is detected as an anomaly.
	'''
	fig, ax = plt.subplots()
	df_daily_jumpsup.plot(legend=False, ax=ax)
	plt.show()
	'''

	# Prepare training data. Get data values from the training
	# timeseries data file and normalize the value data. There is a
	# value for every 5 mins for 14 days.
	# 24 * 60 / 5 = 288timesteps per day
	# 288 * 14 = 4032 data points in total
	# Normalize and save the mean and std.
	training_mean = df_small_noise.mean()
	training_std = df_small_noise.std()
	df_training_value = (df_small_noise - training_mean) / training_std
	print("Number of training samples:", len(df_training_value))

	# Create sequences. Create sequences combining TIME_STEPS
	# contiguous data values from the training data.
	TIME_STEPS = 288

	x_train = create_sequences(df_training_value.values, TIME_STEPS)
	print("Training input shape: ", x_train.shape)

	# Build a model. Build a convolutional reconstruction autoencoder
	# model. The model will take input of shape (batch_size,
	# sequence_length, num_features) and return output of the same
	# shape. In this case, sequence_length is 288 and num_features is
	# 1.
	model = keras.Sequential(
		[
			layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
			layers.Conv1D(
				filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
			),
			layers.Dropout(rate=0.2),
			layers.Conv1D(
				filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
			),
			layers.Conv1DTranspose(
				filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
			),
			layers.Dropout(rate=0.2),
			layers.Conv1DTranspose(
				filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
			),
			layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
		]
	)
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
	model.summary()

	# Train the model. Note that using x_train as both the input and
	# the target since this is a reconstruction model.
	history = model.fit(
		x_train, x_train, epochs=50, batch_size=128, validation_split=0.1,
		callbacks=[
			keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
		],
	)

	# Plot training and validation loss to see how training went.
	'''
	plt.plot(history.history["loss"], label="Training Loss")
	plt.plot(history.history["val_loss"], label="Validation Loss")
	plt.legend()
	plt.show()
	'''

	# Detecting anomalies. Detect anomalies by determining how well the
	# model can reconstruct the input data.
	# 1) Find MAE loss on training samples.
	# 2) Find max MAE loss value. This is the worst the model has
	#	performed trying to reconstruct a sample. This will be the
	#	threshold for anomaly detection.
	# 3) If the reconstruction loss for a sample is greater than this
	#	threshold value then it can be infered that the model is seeing
	#	a patter that it isn't familiar with. This sample will be
	#	labeled an anomaly.
	# Get train MAE loss.
	x_train_pred = model.predict(x_train)
	train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

	'''
	plt.hist(train_mae_loss, bins=50)
	plt.xlabel("Train MAE loss")
	plt.ylabel("No of samples")
	plt.show()
	'''

	# Get reconstruction of loss threshold.
	threshold = np.max(train_mae_loss)
	print("Reconstruction error threshold: ", threshold)

	# Compare reconstruction. Just for fun, see how the model has
	# reconstructed the first sample. This is the 288 timesteps from
	# day 1 of the training dataset.
	# Checking how the first sequence is learned.
	'''
	plt.plot(x_train[0])
	plt.plot(x_train_pred[0])
	plt.show()
	'''

	# Prepare test data.
	def_test_value = (df_daily_jumpsup - training_mean) / training_std
	'''
	fig, ax = plt.subplots()
	def_test_value.plot(legend=False, ax=ax)
	plt.show()
	'''

	# Create sequences from test values.
	x_test = create_sequences(def_test_value.values, TIME_STEPS)
	print("Test input shape: ", x_test.shape)

	# Get test MAE loss.
	x_test_pred = model.predict(x_test)
	test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
	test_mae_loss = test_mae_loss.reshape((-1))

	'''
	plt.hist(test_mae_loss, bins=50)
	plt.xlabel("test MAE loss")
	plt.ylabel("No of samples")
	plt.show()
	'''

	# Detect all the samples which are anomalies.
	anomalies = test_mae_loss > threshold
	print("Number of anomaly samples: ", np.sum(anomalies))
	print("Indices of anomaly samples: ", np.where(anomalies))

	# Plot anomalies. Now that it is known which samples of the data
	# are anomalies, find the corresponding timestamps from the
	# original test data. 
	anomalous_data_indices = []
	for data_idx in range(TIME_STEPS - 1, len(def_test_value) - TIME_STEPS + 1):
		if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
			anomalous_data_indices.append(data_idx)

	# Overlay the anomalies on the original test data plot.
	df_subset = df_daily_jumpsup.iloc[anomalous_data_indices]
	'''
	fig, ax = plt.subplots()
	df_daily_jumpsup.plot(legend=False, ax=ax)
	df_subset.plot(legend=False, ax=ax, color="r")
	plt.show()
	'''

	# Exit the program.
	exit(0)


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=288):
	output = []
	for i in range(len(values) - time_steps):
		output.append(values[i : (i + time_steps)])
	return np.stack(output)


def normalize_test(values, mean, std):
	values -= mean
	values /= std
	return values


if __name__ == '__main__':
	main()