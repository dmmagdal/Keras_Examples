# recommented_transformer.py
# Rating rate prediction using the Behavior Sequence Transformer (BST)
# model on the Movielens.
# This example demonstrates the Behavior Sequence Transformer (BST)
# model, by Qiwei Chen et al., using the Movielens dataset. The BST
# model leverages the sequential behavior of the users in watching and
# rating movies, as well as user profile and movie features, to predict
# the rating of user to a target movie.
# More precisely, the BST model aims to predict the rating of a target
# movie by accepting the following inputs:
# 1) A fixed-length sequence of movie_ids watched by a user.
# 2) A fixed-length sequence of the ratings for the movies watched by a
#	user.
# 3) A set of user features, including user_id, sex, occupation, and 
#	age_group.
# 4) A set of genres for each movie in the input sequence and the 
#	target movie.
# 5) A target_movie_id for which to predict the rating.
# This example modifies the original BST model in the following ways:
# 1) Incorporate the movie features (genres) into the processing of the
#	embedding of each movie of the input sequence and the target movie,
#	rather than treating them as "other features" outside the 
#	transformer layer.
# 2) Utilize the ratings of movies in the input sequence, along with
#	their positions in the sequence, to update them before feeding them
#	into the self-attention layer.
# Note that this example should be run using Tensorflow 2.4 or higher.
# The dataset.
# Using the 1M version of the Movielens dataset. The dataset includes
# around 1 million ratings from 6000 users on 4000 movies, along with
# some user features, movie genres. In addition, the timestamp of each
# user-movie rating is provided, which allows creating sequences of
# movie genres. In addition, the timestamp of each user-movie rating is
# provided, which allows creating sequences of movie ratings for each
# user, as expected by the BST model.
# Source: https://keras.io/examples/structured_data/movielens_
# recommendations_transformers/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import os
import math
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	# Prepare the data.
	# Download and prepare the DataFrames. The downloaded folder will
	# contain three data files: users.dat, movies.dat, and ratings.dat.
	urlretrieve(
		"http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip"
	)
	ZipFile("movielens.zip", "r").extractall()

	# Then load the data into pandas DataFrames with their proper
	# column names.
	users = pd.read_csv(
		"ml-1m/users.dat",
		sep="::",
		names=["user_id", "sex", "age_group", "occupation", "zip_code"],
	)

	ratings = pd.read_csv(
		"ml-1m/ratings.dat",
		sep="::",
		names=["user_id", "movie_id", "rating", "unix_timestamp"],
	)

	movies = pd.read_csv(
		"ml-1m/movies.dat",
		sep="::",
		names=["movie_id", "title", "genres"],
	)

	# Do some simple data processing to fix the data types of the
	# columns.
	users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
	users["age_group"] = users["age_group"].apply(lambda x: f"group_{x}")
	users["occupation"] = users["occupation"].apply(lambda x: f"occupation_{x}")

	movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")

	ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
	ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")
	ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

	# Each movie has multiple genres. Split them into separate columnds
	# in the movies DataFrame.
	genres = [
		"Action",
		"Adventure",
		"Animation",
		"Children's",
		"Comedy",
		"Crime",
		"Documentary",
		"Drama",
		"Fantasy",
		"Film-Noir",
		"Horror",
		"Musical",
		"Mystery",
		"Romance",
		"Sci-Fi",
		"Thriller",
		"War",
		"Western",
	]
	for genre in genres:
		movies[genre] = movies["genres"].apply(
			lambda values: int(genre in values.split("|"))
		)

	# Transform the movie ratings data into sequences.
	# First, sort the ratings data using the unix_timestamp, and then
	# group the movie_id values and the rating values by user_id. The
	# output DataFrame will have a record for each user_id, with two
	# ordered lists (sorted by rating datetime): the movies they have
	# rated, and their ratings of these movies.
	ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

	ratings_data = pd.DataFrame(
		data={
			"user_id": list(ratings_group.groups.keys()),
			"movie_ids": list(ratings_group.movie_id.apply(list)),
			"ratings": list(ratings_group.rating.apply(list)),
			"timestamps": list(ratings_group.unix_timestamp.apply(list)),
		}
	)

	# Now, split the movie_ids list into a set of sequences of a fixed
	# length. Do the same for ratings. Set the sequence_length variable
	# to change the length of the input sequence to the model. Can also
	# change the step_size to control the number of sequences to
	# generate for each user.
	sequence_length = 4
	step_size = 2

	ratings_data.movie_ids = ratings_data.movie_ids.apply(
		lambda ids: create_sequences(ids, sequence_length, step_size)
	)

	ratings_data.ratings = ratings_data.ratings.apply(
		lambda ids: create_sequences(ids, sequence_length, step_size)
	)

	del ratings_data["timestamps"]

	# After that, process the output of each sequence in a separate
	# records in the DataFrame. In addition, join the user features
	# with the ratings data.
	ratings_data_movies = ratings_data[["user_id", "movie_ids"]].explode(
		"movie_ids", ignore_index=True
	)
	ratings_data_rating = ratings_data[["ratings"]].explode(
		"ratings", ignore_index=True
	)
	ratings_data_transformed = pd.concat(
		[ratings_data_movies, ratings_data_rating], axis=1
	)
	ratings_data_transformed = ratings_data_transformed.join(
		users.set_index("user_id"), on="user_id"
	)

	# Side note: Had an issue running the program. Apparently the
	# DataFrame had some NaN values in it which caused problems when
	# applying the lambda functions. Removing these rows with NaN
	# values with this line of code seems to have resolved that issue.
	ratings_data_transformed.dropna(inplace=True)
	
	ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
		lambda x: ",".join(x)
	)
	ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
		lambda x: ",".join([str(v) for v in x])
	)

	del ratings_data_transformed["zip_code"]

	ratings_data_transformed.rename(
		columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
		inplace=True,
	)

	# With sequence_length of 4 and step_size of 2, end up with 498,623
	# sequences. Finally, split the data into training and testing
	# splits, with 85% and 15% of the instances, respectively, and
	# store them to CSV files.
	random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
	train_data = ratings_data_transformed[random_selection]
	test_data = ratings_data_transformed[~random_selection]

	train_data.to_csv("train_data.csv", index=False, sep="|", header=False)
	test_data.to_csv("test_data.csv", index=False, sep="|", header=False)

	# Define metadata.
	CSV_HEADER = list(ratings_data_transformed.columns)

	CATEGORICAL_FEATURES_WITH_VOCABULARY = {
		"user_id": list(users.user_id.unique()),
		"movie_id": list(movies.movie_id.unique()),
		"sex": list(users.sex.unique()),
		"age_group": list(users.age_group.unique()),
		"occupation": list(users.occupation.unique()),
	}

	USER_FEATURES = ["sex", "age_group", "occupation"]

	MOVIE_FEATURES = ["genres"]

	include_user_id = False
	include_user_features = False
	include_movie_features = False

	hidden_units = [256, 128]
	dropout_rate = 0.1
	num_heads = 3

	model = create_model(
		sequence_length, USER_FEATURES, CATEGORICAL_FEATURES_WITH_VOCABULARY, 
		include_user_id, include_user_features, include_movie_features, num_heads, 
		dropout_rate, movies, genres, hidden_units
	)

	# Run training and evaluation experiment.
	# Compile the model.
	model.compile(
		optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
		loss=keras.losses.MeanSquaredError(),
		metrics=[keras.metrics.MeanAbsoluteError()],
	)

	# Read the training data.
	train_dataset = get_dataset_from_csv(
		"train_data.csv", CSV_HEADER, shuffle=True, batch_size=256
	)

	# Fit the model with the training data.
	model.fit(train_dataset, epochs=5)

	# Read the test data.
	test_dataset = get_dataset_from_csv(
		"test_data.csv", CSV_HEADER, shuffle=False, batch_size=256
	)

	# Evaluate the model on the test data.
	_, rmse = model.evaluate(test_dataset, verbose=0)
	print(f"Test MAE: {round(rmse, 3)}")

	# Exit the program.
	exit(0)


def create_sequences(values, window_size, step_size):
	sequences = []
	start_index = 0
	while True:
		end_index = start_index + window_size
		seq = values[start_index:end_index]
		if len(seq) < window_size:
			seq = values[-window_size:]
			if len(seq) == window_size:
				sequences.append(seq)
			break
		sequences.append(seq)
		start_index += step_size
	return sequences


# Create tf.data.Dataset for training and evaluation.
def get_dataset_from_csv(csv_file_path, CSV_HEADER, shuffle=False, batch_size=128):
	def process(features):
		movie_ids_string = features["sequence_movie_ids"]
		sequence_movie_ids = tf.strings.split(movie_ids_string, ",").to_tensor()

		# The last movie id in the sequence is the target movie.
		features["target_movie_id"] = sequence_movie_ids[:, -1]
		features["sequence_movie_ids"] = sequence_movie_ids[:, :-1]

		ratings_string = features["sequence_ratings"]
		sequence_ratings = tf.strings.to_number(
			tf.strings.split(ratings_string, ","), tf.dtypes.float32
		).to_tensor()

		# The last rating in the sequence is the target for the model
		# to predict.
		target = sequence_ratings[:, -1]
		features["sequence_ratings"] = sequence_ratings[:, :-1]

		return features, target

	dataset = tf.data.experimental.make_csv_dataset(
		csv_file_path,
		batch_size=batch_size,
		column_names=CSV_HEADER,
		num_epochs=1,
		header=False,
		field_delim="|",
		shuffle=shuffle,
	).map(process)

	return dataset


# Create model inputs.
def create_model_inputs(sequence_length):
	return {
		"user_id": layers.Input(name="user_id", shape=(1,), dtype=tf.string),
		"sequence_movie_ids": layers.Input(
			name="sequence_movie_ids", shape=(sequence_length - 1,), dtype=tf.string 
		),
		"target_movie_id": layers.Input(
			name="target_movie_id", shape=(1,), dtype=tf.string
		),
		"sequence_ratings": layers.Input(
			name="sequence_ratings", shape=(sequence_length - 1,), dtype=tf.float32
		),
		"sex": layers.Input(name="sex", shape=(1,), dtype=tf.string),
		"age_group": layers.Input(name="age_group", shape=(1,), dtype=tf.string),
		"occupation": layers.Input(name="occupation", shape=(1,), dtype=tf.string),
	}


# Encode input features.
# The encode_input_features method works as follows:
# 1) Each categorical user feature is encoded using layers.Embedding,
#	with embedding dimension equal to the square root of the vocabulary
#	size of the feature. The embeddings of these features are
#	concatenated to form a single input tensor.
# 2) Each movie in the movie sequence and the target movie is encoded
#	layers.Embedding, where the dimension size is the square root of
#	the number of movies.
# 3) A multi-hot genres vector for each movie is concatenated with its
#	embedding vector, and processed using a non-linear layers.Dense to
#	output a vector of the same movie embedding dimensions.
# 4) A positional embedding is added to each movie embedding in 
#	sequence, and then multiplied by its rating from the ratings
#	sequence.
# 5) The targe movie embedding is concatenated to the sequence movie 
#	embeddings, producing a tensor with the shape [batch_size, 
#	sequence_length, embedding_size], as expected by the attention 
#	layer for the transformer architecture.
# 6) The method returns a tuple of two elements: 
#	encoded_transformer_features and encoded_other_features.
def encode_input_features(inputs, sequence_length, USER_FEATURES, 
		CATEGORICAL_FEATURES_WITH_VOCABULARY, movies, genres, include_user_id=True, 
		include_user_features=True, include_movie_features=True):
	encoded_transformer_features = []
	encoded_other_features = []

	other_feature_names = []
	if include_user_id:
		other_feature_names.append("user_id")
	if include_movie_features:
		other_feature_names.extend(USER_FEATURES)

	# Encode user features.
	for feature_name in other_feature_names:
		# Conver the string input values into integer indices.
		vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
		idx = StringLookup(
			vocabulary=vocabulary, mask_token=None, num_oov_indices=0
		)(inputs[feature_name])

		# Compute embedding dimensions.
		embedding_dims = int(math.sqrt(len(vocabulary)))

		# Create an embedding layer with the specified dimensions.
		embedding_encoder = layers.Embedding(
			input_dim=len(vocabulary),
			output_dim=embedding_dims,
			name=f"{feature_name}_embedding",
		)

		# Convert the index values to embedding representations.
		encoded_other_features.append(embedding_encoder(idx))

	# Create a single embedding vector for the user features.
	if len(encoded_other_features) > 1:
		encoded_other_features = layers.concatenate(encoded_other_features)
	elif len(encoded_other_features) == 1:
		encoded_other_features = encoded_other_features[0]
	else:
		encoded_other_features = None

	# Create a movie embedding encoder.
	movie_vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY["movie_id"]
	movie_embedding_dims = int(math.sqrt(len(movie_vocabulary)))

	# Create a lookup to convert string values to integer indices.
	movie_index_lookup = StringLookup(
		input_dim=len(movie_vocabulary),
		mask_token=None,
		num_oov_indices=0,
		name="movie_index_lookup",
	)

	# Create an embedding layer with the specified dimensions.
	movie_embedding_encoder = layers.Embedding(
		input_dim=len(movie_vocabulary),
		output_dim=movie_embedding_dims,
		name=f"movie_embedding",
	)

	# Create a vector lookup for movie genres.
	genre_vectors = movies[genres].to_numpy()
	movie_genres_lookup = layers.Embedding(
		input_dim=genre_vectors.shape[0],
		output_dim=genre_vectors.shape[1],
		embeddings_initializer=tf.keras.initializers.Constant(genre_vectors),
		trainable=False,
		name="genres_vector"
	)

	# Create a processing layer for genres.
	movie_embedding_processor = layers.Dense(
		units=movie_embedding_dims,
		activation="relu",
		name="process_movie_embedding_with_genres",
	)

	# Define a function to encode a given movie id.
	def encode_movie(movie_id):
		# Convert the string input values into integer indices.	
		movie_idx = movie_index_lookup(movie_id)
		movie_embedding = movie_embedding_encoder(movie_idx)
		encoded_movie = movie_embedding
		if include_movie_features:
			movie_genres_vector = movie_genres_lookup(movie_idx)
			encoded_movie = movie_embedding_processor(
				layers.concatenate([movie_embedding, movie_genres_vector])
			)
		return encoded_movie

	# Encoded target_movie_id.
	target_movie_id = inputs["target_movie_id"]
	encoded_target_movie = encode_movie(target_movie_id)

	# Encoding sequence movie_ids.
	sequence_movie_ids = inputs["sequence_movie_ids"]
	encoded_sequence_movies = encode_movie(sequence_movie_ids)

	# Create positional embedding.
	positional_embedding_encoder = layers.Embedding(
		input_dim=sequence_length,
		output_dim=movie_embedding_dims,
		name="positional_embedding",
	)
	positions = tf.range(start=0, limit=sequence_length - 1, delta=1)
	encoded_positions = positional_embedding_encoder(positions)

	# Retrieve sequence ratings to incorporate them into the encoding
	# of the movie.
	sequence_ratings = tf.expand_dims(inputs["sequence_ratings"], -1)

	# Add the positional encoding to the movie encodings and multiply
	# them by rating.
	encoded_sequence_movies_with_position_and_rating = layers.Multiply()(
		[(encoded_sequence_movies + encoded_positions), sequence_ratings]
	)

	# Construct the transformer inputs.
	for encoded_movie in tf.unstack(
		encoded_sequence_movies_with_position_and_rating, axis=1
	):
		encoded_transformer_features.append(tf.expand_dims(encoded_movie, 1))
	encoded_transformer_features.append(encoded_target_movie)

	encoded_transformer_features = layers.concatenate(
		encoded_transformer_features, axis=1
	)

	return encoded_transformer_features, encoded_other_features


# Create a BST model.
def create_model(sequence_length, USER_FEATURES, CATEGORICAL_FEATURES_WITH_VOCABULARY, 
		include_user_id, include_user_features, include_movie_features, num_heads, 
		dropout_rate, movies, genres, hidden_units):
	inputs = create_model_inputs(sequence_length)
	transformer_features, other_features = encode_input_features(
		inputs, sequence_length, USER_FEATURES, 
		CATEGORICAL_FEATURES_WITH_VOCABULARY, movies, genres, include_user_id, 
		include_user_features, include_movie_features
	)

	# Create a multi-head attention layer.
	attention_output = layers.MultiHeadAttention(
		num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
	)(transformer_features, transformer_features)

	# Transformer block.
	attention_output = layers.Dropout(dropout_rate)(attention_output)
	x1 = layers.Add()([transformer_features, attention_output])
	x1 = layers.LayerNormalization()(x1)
	x2 = layers.LeakyReLU()(x1)
	x2 = layers.Dense(units=x2.shape[-1])(x2)
	x2 = layers.Dropout(dropout_rate)(x2)
	transformer_features = layers.Add()([x1, x2])
	transformer_features = layers.LayerNormalization()(transformer_features)
	features = layers.Flatten()(transformer_features)

	# Included the other features.
	if other_features is not None:
		features = layers.concatenate(
			[features, layers.Reshape([other_features.shape[-1]])(other_features)]
		)

	# Fully-connected layers.
	for num_units in hidden_units:
		features = layers.Dense(num_units)(features)
		features = layers.BatchNormalization()(features)
		features = layers.LeakyReLU()(features)
		features = layers.Dropout(dropout_rate)(features)

	outputs = layers.Dense(units=1)(features)
	model = keras.Model(inputs=inputs, outputs=outputs)
	return model


if __name__ == '__main__':
	main()