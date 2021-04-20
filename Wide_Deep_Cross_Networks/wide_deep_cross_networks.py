# wide_deep_cross_networks.py
# Using Wide & Deep and Deep & Cross networks for structured data
# classification.
# This example uses the Covertype dataset from the UCI Machine Learning
# Repository. The task is to predict forest cover type from
# cartographic variables. The dataset includes 506,011 instances with
# 12 input features: 10 numerical features and 2 categorical features.
# Each instance is categorized into 1 of 7 classes.
# Source: https://keras.io/examples/structured_data/wide_deep_cross_
# networks/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


def main():
	# Prepare the data.
	# Load the dataset from the UCI Machine Learning Repository into a
	# pandas DataFrame.
	data_url = (
		"https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
	)
	raw_data = pd.read_csv(data_url, header=None)
	print(f"Dataset shape: {raw_data.shape}")
	raw_data.head()

	# The two categorical features in the dataset are binary-encoded.
	# Convert this dataset representation to the typical
	# representation, where each categorical feature is represented as
	# a single interger value.
	soil_type_values = [f"soil_type_{idx + 1}" for idx in range(40)]
	wilderness_area_values = [f"area_type_{idx + 1}" for idx in range(4)]

	soil_type = raw_data.loc[:, 14:53].apply(
		lambda x: soil_type_values[0::1][x.to_numpy().nonzero()[0][0]], axis=1
	)
	wilderness_area = raw_data.loc[:, 10:13].apply(
		lambda x: wilderness_area_values[0::1][x.to_numpy().nonzero()[0][0]], axis=1
	)

	CSV_HEADER = [
		"Elevation",
		"Aspect",
		"Slope",
		"Horizontal_Distance_To_Hydrology",
		"Vertical_Distance_To_Hydrology",
		"Horizontal_Distance_To_Roadways",
		"Hillshade_9am",
		"Hillshade_Noon",
		"Hillshade_3pm",
		"Horizontal_Distance_To_Fire_Points",
		"Wilderness_Area",
		"Soil_Type",
		"Cover_Type",
	]

	data = pd.concat(
		[raw_data.loc[:, 0:9], wilderness_area, soil_type, raw_data.loc[:, 54]],
		axis=1,
		ignore_index=True,
	)
	data.columns = CSV_HEADER

	# Convert the target label indices into a range from 0 to 6 (there
	# are 7 labels in total).
	data["Cover_Type"] = data["Cover_Type"] - 1

	print(f"Dataset shape: {data.shape}")
	data.head().T

	# The shape of the DataFrame shows there are 13 columns per sample
	# (12 for the features and 1 for the target label).
	# Split the data into training (85%) and test (15%) sets.
	train_splits = []
	test_splits = []

	for _, group_data in data.groupby("Cover_Type"):
		random_selection = np.random.rand(len(group_data.index)) <= 0.85
		train_splits.append(group_data[random_selection])
		test_splits.append(group_data[~random_selection])

	train_data = pd.concat(train_splits).sample(frac=1).reset_index(drop=True)
	test_data = pd.concat(test_splits).sample(frac=1).reset_index(drop=True)

	print(f"Train split size: {len(train_data.index)}")
	print(f"Test split size: {len(test_data.index)}")

	# Next, store the training and test data in separate CSV files.
	train_data_file = "train_data.csv"
	test_data_file = "test_data.csv"

	train_data.to_csv(train_data_file, index=False)
	test_data.to_csv(test_data_file, index=False)

	# Define dataset metadata.
	# Define the metadata of the dataset that will be useful for
	# reading and parsing the data into input features, and encoding
	# the input features with respect to their types.
	TARGET_FEATURE_NAME = "Cover_Type"

	TARGET_FEATURE_LABELS = ["0", "1", "2", "3", "4", "5", "6"]

	NUMERIC_FEATURE_NAMES = [
		"Aspect",
		"Elevation",
		"Hillshade_3pm",
		"Hillshade_9am",
		"Hillshade_Noon",
		"Horizontal_Distance_To_Fire_Points",
		"Horizontal_Distance_To_Hydrology",
		"Horizontal_Distance_To_Roadways",
		"Slope",
		"Vertical_Distance_To_Hydrology",
	]

	CATEGORICAL_FEATURES_WITH_VOCABULARY = {
		"Soil_Type": list(data["Soil_Type"].unique()),
		"Wilderness_Area": list(data["Wilderness_Area"].unique()),
	}

	CATEGORICAL_FEATURES_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

	FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURES_NAMES

	COLUMN_DEFAULTS = [
		[0] if feature_name in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME] else ["NA"]
		for feature_name in CSV_HEADER
	]

	NUM_CLASSES = len(TARGET_FEATURE_LABELS)

	# Experiment setup.
	# Configure the parameters and implement the procedure for running
	# a training and evaluation experiment given a model.
	learning_rate = 0.001
	dropout_rate = 0.1
	batch_size = 265
	num_epochs = 50

	hidden_units = [32, 32]

	baseline_model = create_baseline_model(
		FEATURE_NAMES, NUMERIC_FEATURE_NAMES, CATEGORICAL_FEATURES_NAMES, 
		CATEGORICAL_FEATURES_WITH_VOCABULARY, hidden_units, dropout_rate, 
		NUM_CLASSES
	)
	#keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")

	# Run the experiment with the baseline model.
	run_experiment(
		baseline_model, learning_rate, num_epochs, batch_size, CSV_HEADER,
		COLUMN_DEFAULTS, TARGET_FEATURE_NAME, train_data_file, test_data_file
	)

	# The baseline linear model achieves ~76% test accuracy.

	wide_and_deep_model = create_wide_and_deep_model(
		FEATURE_NAMES, NUMERIC_FEATURE_NAMES, CATEGORICAL_FEATURES_NAMES, 
		CATEGORICAL_FEATURES_WITH_VOCABULARY, hidden_units, dropout_rate, 
		NUM_CLASSES
	)
	#keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")

	# Run the experiment with the wide and deep model.
	run_experiment(
		wide_and_deep_model, learning_rate, num_epochs, batch_size, 
		CSV_HEADER, COLUMN_DEFAULTS, TARGET_FEATURE_NAME, train_data_file, 
		test_data_file
	)

	# The wide and deep model achieves ~79% test accuracy.

	deep_and_cross = create_deep_and_cross_model(
		FEATURE_NAMES, NUMERIC_FEATURE_NAMES, CATEGORICAL_FEATURES_NAMES, 
		CATEGORICAL_FEATURES_WITH_VOCABULARY, hidden_units, dropout_rate, 
		NUM_CLASSES
	)
	#keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")

	# Run the experiment with the deep and cross model.
	run_experiment(
		wide_and_deep_model, learning_rate, num_epochs, batch_size, 
		CSV_HEADER, COLUMN_DEFAULTS, TARGET_FEATURE_NAME, train_data_file, 
		test_data_file
	)

	# The wide and deep model achieves ~81% test accuracy.

	# Conclusion.
	# Keras preprocessing layers can be used to easily handle
	# categorical features with different encoding mechanisms,
	# including one-hot encoding and feature embedding. In addition,
	# different model architectures - like wide, deep, and cross
	# networks - have different advantages, with respect to different
	# dataset properties. You can explore using them independently or
	# combining them to achieve the best result for your dataset.

	# Exit the program.
	exit(0)


# Define an input function that reads and parses the file, then
# converts features and labels into a tf.data.Dataset for training or
# evaluation.
def get_dataset_from_csv(csv_file_path, batch_size, CSV_HEADER, 
	COLUMN_DEFAULTS, TARGET_FEATURE_NAME, shuffle=False):
	dataset = tf.data.experimental.make_csv_dataset(
		csv_file_path, batch_size=batch_size, column_names=CSV_HEADER,
		column_defaults=COLUMN_DEFAULTS, label_name=TARGET_FEATURE_NAME,
		num_epochs=1, header=True, shuffle=shuffle
	)
	return dataset.cache()


def run_experiment(model, learning_rate, num_epochs, batch_size, CSV_HEADER,
		COLUMN_DEFAULTS, TARGET_FEATURE_NAME, train_data_file, test_data_file):
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
		loss=keras.losses.SparseCategoricalCrossentropy(),
		metrics=[keras.metrics.SparseCategoricalAccuracy()],
	)

	train_dataset = get_dataset_from_csv(
		train_data_file, batch_size, CSV_HEADER, COLUMN_DEFAULTS,
		TARGET_FEATURE_NAME, shuffle=True
	)

	test_dataset = get_dataset_from_csv(
		test_data_file, batch_size, CSV_HEADER, COLUMN_DEFAULTS,
		TARGET_FEATURE_NAME, shuffle=False 
	)

	print("Start training the model...")
	history = model.fit(train_dataset, epochs=num_epochs)
	print("Model training finished.")

	_, accuracy = model.evaluate(test_dataset, verbose=0)

	print(f"Test accuracy: {round(accuracy * 100, 2)}%")


# Create model inputs.
# Define the inputs for the models as a dictionary, where the key is
# the feature name, and the value is a keras.layers.Input tensor with
# the corresponding feature shape and data type.
def create_model_inputs(FEATURE_NAMES, NUMERIC_FEATURE_NAMES):
	inputs = {}
	for feature_name in FEATURE_NAMES:
		if feature_name in NUMERIC_FEATURE_NAMES:
			inputs[feature_name] = layers.Input(
				name=feature_name, shape=(), dtype=tf.float32
			)
		else:
			inputs[feature_name] = layers.Input(
				name=feature_name, shape=(), dtype=tf.string
			)
	return inputs


# Encode features.
# Create two representations of the input featuers: sparse and dense:
# 1. In sparse representation, the categorical features are encoded
# with one-hot encoding using the CategoryEncoding layer. This
# representation can be useful for the model to memorize particular
# feature values to make certain predictions.
# 2. In the dense representation, the categorical features are encoded
# with low-dimensional embeddings using the Embedding layer. This
# representaton helps the model to generalize well to unseen feature
# combinations.
def encode_inputs(inputs, CATEGORICAL_FEATURES_NAMES,
		CATEGORICAL_FEATURES_WITH_VOCABULARY, use_embedding=False):
	encoded_features = []
	for feature_name in inputs:
		if feature_name in CATEGORICAL_FEATURES_NAMES:
			vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]

			# Create a lookup to convert string values to an integer
			# indices. Since no using a mask token nor expecting any
			# out of vocabulary (oov) token, set mask_token to None and
			# num_oov_indices to 0.
			index = StringLookup(
				vocabulary=vocabulary, mask_token=None, num_oov_indices=0
			)

			# Convert the string input values into integer indices.
			value_index = index(inputs[feature_name])
			if use_embedding:
				embedding_dims = int(math.sqrt(len(vocabulary)))

				# Create an embedding layer with the specified
				# dimensions.
				embedding_encoder = layers.Embedding(
					input_dim=len(vocabulary), output_dim=embedding_dims
				)

				# Convert the index values to embedding representaion.
				encoded_feature = embedding_encoder(value_index)
			else:
				# Create a one-hot encoder.
				onehot_encoder = CategoryEncoding(output_mode="binary")
				onehot_encoder.adapt(index(vocabulary))

				# Convert the index values to a one-hot representation.
				encoded_feature = onehot_encoder(value_index)
		else:
			# Use the numerical features as-is.
			encoded_feature = tf.expand_dims(inputs[feature_name], -1)

		encoded_features.append(encoded_feature)

	all_features = layers.concatenate(encoded_features)
	return all_features


# Experimenta 1: A baseline model.
# In the first experiment, create a multi-layer feed-forward netweork,
# where the categorical features are one-hot encoded.
def create_baseline_model(FEATURE_NAMES, NUMERIC_FEATURE_NAMES,
		CATEGORICAL_FEATURES_NAMES, CATEGORICAL_FEATURES_WITH_VOCABULARY,
		hidden_units, dropout_rate, NUM_CLASSES):
	inputs = create_model_inputs(FEATURE_NAMES, NUMERIC_FEATURE_NAMES)
	features = encode_inputs(
		inputs, CATEGORICAL_FEATURES_NAMES,	CATEGORICAL_FEATURES_WITH_VOCABULARY, 
		use_embedding=False
	)

	for units in hidden_units:
		features = layers.Dense(units)(features)
		features = layers.BatchNormalization()(features)
		features = layers.ReLU()(features)
		features = layers.Dropout(dropout_rate)(features)

	outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(features)
	model = keras.Model(inputs=inputs, outputs=outputs)
	return model


# Experiment 2: Wide & Deep model.
# In the second experiment, create a wide and deep model. The wide part
# of the model a linear mode, while the deep part of the model is a
# multi-layer feed-forward network.
# Use sparse representation of the input features in the wide part of
# the model and the dense representation of the input features for the
# deep pare of the model.
# Note that every input features contributes to both parts of the model
# with different representations.
def create_wide_and_deep_model(FEATURE_NAMES, NUMERIC_FEATURE_NAMES,
		CATEGORICAL_FEATURES_NAMES, CATEGORICAL_FEATURES_WITH_VOCABULARY,
		hidden_units, dropout_rate, NUM_CLASSES):
	inputs = create_model_inputs(FEATURE_NAMES, NUMERIC_FEATURE_NAMES)
	wide = encode_inputs(
		inputs, CATEGORICAL_FEATURES_NAMES,	CATEGORICAL_FEATURES_WITH_VOCABULARY,
		use_embedding=False
	)
	wide = layers.BatchNormalization()(wide)

	deep = encode_inputs(
		inputs, CATEGORICAL_FEATURES_NAMES,	CATEGORICAL_FEATURES_WITH_VOCABULARY,
		use_embedding=True
	)
	for units in hidden_units:
		deep = layers.Dense(units)(deep)
		deep = layers.BatchNormalization()(deep)
		deep = layers.ReLU()(deep)
		deep = layers.Dropout(dropout_rate)(deep)

	merged = layers.concatenate([wide, deep])
	outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(merged)
	model = keras.Model(inputs=inputs, outputs=outputs)
	return model


# Experiment 3: Deep & Cross model.
# In the third experiment, create a deep and cross mode. The deep part
# of the model is the same as the deep part created in the previous
# experiment. The key idea of the cross part is to apply explicit
# feature crossing in an efficient way, where the degree of the cross
# features grows with layer depth.
def create_deep_and_cross_model(FEATURE_NAMES, NUMERIC_FEATURE_NAMES,
		CATEGORICAL_FEATURES_NAMES, CATEGORICAL_FEATURES_WITH_VOCABULARY,
		hidden_units, dropout_rate, NUM_CLASSES):
	inputs = create_model_inputs(FEATURE_NAMES, NUMERIC_FEATURE_NAMES)
	x0 = encode_inputs(inputs, CATEGORICAL_FEATURES_NAMES,
		CATEGORICAL_FEATURES_WITH_VOCABULARY, use_embedding=True
	)

	cross = x0
	for _ in hidden_units:
		units = cross.shape[-1]
		x = layers.Dense(units)(cross)
		cross = x0 * x + cross
	cross = layers.BatchNormalization()(cross)

	deep = x0
	for units in hidden_units:
		deep = layers.Dense(units)(deep)
		deep = layers.BatchNormalization()(deep)
		deep = layers.ReLU()(deep)
		deep = layers.Dropout(dropout_rate)(deep)

	merged = layers.concatenate([cross, deep])
	outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(merged)
	model = keras.Model(inputs=inputs, outputs=outputs)
	return model


if __name__ == '__main__':
	main()