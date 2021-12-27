# multi-label_class.py
# Implementing a large-scale multi-label text classifcation model.
# raining a VQ-VAE for the image reconstruction and codebook sampling
# for generation.
# Introduction
# This example will build a multi-label text classifier to predict the
# subject areas of arXiv papers from their abstract bodies. This type
# of classifier can be useful for conference submission portals like
# OpenReview. Given a paper abstract, the portal could provide
# suggestions for which areas the paper would best belong to.
# The dataset was collected using th arXiv Python library that provides
# a wrapper around the original arXiv API. To learn more about the data
# collection process, please refer to this notebook. Additionally, the
# dataset could be found on Kaggle.
# Source: https://keras.io/examples/nlp/multi_label_classification/
# Tensorflow 2.6.0
# Python 3.7
# Windows/MacOS/Linux


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from ast import literal_eval
#import matplotlib.pyplot as plt


def main():
	# Perform exploratory data analysis
	# In this section, first load the dataset into a pandas dataframe
	# and then perform some basic exploratory data analysis (EDA).
	arxiv_data = pd.read_csv(
		"https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv"
	)
	arxiv_data.head()

	# The text features are present in the summaries column and their
	# corresponding labels are in terms. There are multiple categories
	# associated with a particular entry.
	print(f"Therea rea {len(arxiv_data)} rows in the dataset.")

	# Real-word data is noisy. One of the most commonly observed source
	# of noise is data duplication. Here, notice the initial dataset
	# has got about 13K duplicate entries.
	total_duplicate_titles = sum(arxiv_data["titles"].duplicated())
	print(f"There are {total_duplicate_titles} duplicate titles.")

	# Before proceeding further, drop these entries.
	arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]
	print(f"There are {len(arxiv_data)} rows in the deduplicated dataset.")

	# There are some terms with occurence as low as 1.
	print(sum(arxiv_data["terms"].value_counts() == 1))

	# How many unique terms?
	print(arxiv_data["terms"].unique())

	# As observed above, out of 3,157 unique combinations of terms,
	# 2,321 entries have the lowest occurence. to prepare the train,
	# validation, and test sets with stratification, must drop these
	# terms.
	# Filtering the rare terms.
	arxiv_data_filtered = arxiv_data.groupby("terms").filter(
		lambda x: len(x) > 1
	)
	print(arxiv_data_filtered.shape)

	# Convert the string labels to a list of strings.
	# The initial labels are represented as raw strings. Here, make
	# them List[str] for a more compact representation.
	arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(
		lambda x: literal_eval(x)
	)
	print(arxiv_data_filtered["terms"].values[:5])

	# Use stratified splits because of class imbalance.
	# The dataset has a class imabalance problem. So, to have a fair
	# evaluation result, ensure the datasets are sampled with
	# stratification. To know more about different strategies to deal
	# with the class imbalance problem, follow this tutorial. For an
	# end-to-end demonstration of classifcation with imbalanced data,
	# refer to imbalanced classifcation: credit card fraud detection.
	test_split = 0.1

	# Initial train and test split.
	train_df, test_df = train_test_split(
		arxiv_data_filtered, test_size=test_split,
		stratify=arxiv_data_filtered["terms"].values,
	)

	# Splitting the test set further into validation and new test sets.
	val_df = test_df.sample(frac=0.5)
	test_df.drop(val_df.index, inplace=True)

	print(f"Number of rows in training set: {len(train_df)}")
	print(f"Number of rows in validation set: {len(val_df)}")
	print(f"Number of rows in test set: {len(test_df)}")

	# Multi-label binarization.
	# Preprocess the labels using the StringLookup layer.
	terms = tf.ragged.constant(train_df["terms"].values)
	lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
	lookup.adapt(terms)
	vocab = lookup.get_vocabulary()

	def invert_multi_hot(encoded_labels):
		# Reverse a single multi-hot encoded label to a tuple of vocab
		# terms.
		hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
		return np.take(vocab, hot_indices)

	print("Vocabulary:\n")
	print(vocab)

	# Here, separate the individual unique classes available from the
	# label pool and then using this information to represent a given
	# label set with 0's and 1's.
	sample_label = train_df["terms"].iloc[0]
	print(f"Original label: {sample_label}")

	label_binarized = lookup([sample_label])
	print(f"Label-binarized representation: {label_binarized}")

	# Data preprocessing and tf.data.Dataset objects.
	# First, get the percentile estimates of the sequence lengths. The
	# purpose will be clear in a moment.
	train_df["summaries"].apply(lambda x: len(x.split(" "))).describe()

	# Notice that 50% of the abstracts have a length of 154 (This
	# number may be different based upon the split). So, any number
	# close to that value is a good enough approximate for the
	# maximum sequence length.
	# Now, implement utilities to prepare the datasets that would go
	# straight to the text classifier model.
	max_seqlen = 150
	batch_size = 128
	padding_token = "<pad>"
	auto = tf.data.AUTOTUNE

	def unify_text_length(text, label):
		# Split the given abstract and calculate its length.
		word_splits = tf.strings.split(text, sep=" ")
		sequence_length = tf.shape(word_splits)[0]

		# Calculate the padding amount.
		padding_amount = max_seqlen - sequence_length

		# Check if there needs to be a pad or truncate.
		if padding_amount > 0:
			unified_text = tf.pad(
				[text], [[0, padding_amount]], constant_values="<pad>"
			)
			unified_text = tf.strings.reduce_join(
				unified_text, separator=""
			)
		else:
			unified_text = tf.strings.reduce_join(
				word_splits[:max_seqlen], separator=" "
			)

		# The expansion is needed for subsequent vectorization.
		return tf.expand_dims(unified_text, -1), label

	def make_dataset(dataframe, is_train=True):
		labels = tf.ragged.constant(dataframe["terms"].values)
		label_binarized = lookup(labels)
		dataset = tf.data.Dataset.from_tensor_slices(
			(dataframe["summaries"].values, label_binarized)
		)
		dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
		dataset = dataset.map(unify_text_length, num_parallel_calls=auto).cache()
		return dataset.batch(batch_size)

	# Can now prepare the tf.data.Dataset objects.
	train_dataset = make_dataset(train_df, is_train=True)
	validation_dataset = make_dataset(val_df, is_train=False)
	test_dataset = make_dataset(test_df, is_train=False)

	# Dataset preview.
	text_batch, label_batch = next(iter(train_dataset))

	for i, text in enumerate(text_batch[:5]):
		label = label_batch[i].numpy()[None, ...]
		print(f"Abstract: {text[0]}")
		print(f"Label(s): {invert_multi_hot(label[0])}")
		print(" ")

	# Vectorization.
	# Before feeding the data to the model, it must be vectorized
	# (represented in a numerical form). For that purpose, use the
	# TextVectorization layer. It can operate as part of the main model
	# so that the model is excluded from the core processing logic.
	# This greatly reduces the chances of training/serving skew during
	# inference.
	# First, calculate the number of unique words present in the
	# abstracts.
	train_df["total_words"] = train_df["summaries"].str.split().str.len()
	vocabulary_size = train_df["total_words"].max()
	print("Vocabulary size: {vocabulary_size}")

	# Now create the vectorization layer and map() to the
	# tf.data.Datasets created earlier.
	text_vectorizer = layers.TextVectorization(
		max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
	)

	# "TextVectorization" layer needs to be adapted as per the
	# vocabulary from the training set.
	with tf.device("/CPU:0"):
		text_vectorizer.adapt(
			train_dataset.map(lambda text, label: text)
		)

	train_dataset = train_dataset.map(
		lambda text, label: (text_vectorizer(text), label), 
		num_parallel_calls=auto
	).prefetch(auto)
	validation_dataset = validation_dataset.map(
		lambda text, label: (text_vectorizer(text), label), 
		num_parallel_calls=auto
	).prefetch(auto)
	test_dataset = test_dataset.map(
		lambda text, label: (text_vectorizer(text), label), 
		num_parallel_calls=auto
	).prefetch(auto)

	# A batch of raw text will first go through the TextVectorization
	# layer and it will generate their integer representations.
	# Internally, the TextVectorization layer will first create
	# bi-grams out of the sequences and then represent them using 
	# TF-IDF. The output representation will then be passed to the
	# shallow model responsible for text classification.
	# To learn more about other possible configurations with
	# TextVectorizer, please consult the official documentation.
	# Note: Setting the max_tokens argument to a pre-calculated
	# vocabulary size is not a requirement.

	# Create a text classification model.
	# Keep the model simple -- it will be a small stack of fully-
	# connected layers with ReLU as the non-linearity.
	def make_model():
		shallow_mlp_model = keras.Sequential(
			[
				layers.Dense(512, activation="relu"),
				layers.Dense(256, activation="relu"),
				layers.Dense(
					lookup.vocabulary_size(), activation="sigmoid"
				), # More on why "sigmoid" has been used here in a moment.
			]
		)
		return shallow_mlp_model

	# Train the model.
	# Train the model using binary crossentropy loss. This is because
	# the labels are not disjoint. For a given abstract, there may be
	# multiple categories. So, the prediction task will be divided into
	# a series of multiple binary classification problems. This is also
	# why the activation function of the classification layer was kept
	# to sigmoid. Researchers have used other combinations of loss
	# functions and activation functions as well. For example, in
	# Exporing the Limits of Weakly Supervised Pretraining, Mahajan et
	# al. used the softmax activation function and cross-entropy loss
	# to train their models.
	epochs = 20

	shallow_mlp_model = make_model()
	shallow_mlp_model.compile(
		loss="binary_crossentropy", optimizer="adam",
		metrics=["categorical_accuracy"]
	)

	history = shallow_mlp_model.fit(
		train_dataset, validation_data=validation_dataset, epochs=epochs
	)

	'''
	def plot_results(item):
		plt.plot(history.history[item], label=item)
		plt.plot(history.history["val_" + item], label="val_" + item)
		plt.xlabel("Epochs")
		plt.ylabel(item)
		plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
		plt.legend()
		plt.grid()
		plt.show()

	plot_results("loss")
	plot_results("categorical_accuracy")
	# While training, notice an initial sharp fall in the loss
	# followed by a gradual decay.
	'''

	# Evaluate the model.
	_, categorical_acc = shallow_mlp_model.evaluate(test_dataset)
	print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")

	# The trained model gives an evaluation accuracy of ~87%.

	# Inference.
	# An important feature of the preprocessing layers provided by
	# Keras is that they can be included inside a tf.keras.Model. This
	# example will export an inference model by including the
	# text_vectorization layer on top of shallow_mlp_model. This will
	# allow the inference model to directly operate on raw strings.
	# Note that during training, it is always preferable to use these
	# preprocessing layers as part of the data input pipeline rather
	# than the model to avoid surfacing bottlenecks for the hardware
	# accelerators. This also allows for asynchronous data processing.
	# Create a model for inference.
	model_for_inference = keras.Sequential(
		[text_vectorizer, shallow_mlp_model]
	)

	# Create a small dataset just for demoing inference.
	inference_dataset = make_dataset(
		test_df.sample(100), is_train=False
	)
	text_batch, label_batch = next(iter(inference_dataset))
	predicted_probabilities = model_for_inference.predict(text_batch)

	# Perform inference.
	for i, text in enumerate(text_batch[:5]):
		label = label_batch[i].numpy()[None, ...]
		print(f"Abstract: {text[0]}")
		print(f"Label(s): {invert_multi_hot(label[0])}")
		predicted_probs = [probs for probs in predicted_probabilities[i]]
		top_3_labels = [
			x
			for _, x in sorted(
				zip(predicted_probabilities[i], lookup.get_vocabulary()),
				key=lambda pair: pair[0],
				reverse=True
			)
		][:3]
		print(f"Predicted Label(s): ({', '.join([label for label in top_3_labels])})")
		print(" ")

	# The prediction results are not that great but not below the par
	# for a simple model like the one above. Can improve this
	# performance with models that consider word order like LSTM or
	# even those that use Transformers (Vaswani et al.).

	# Exit the program.
	exit(0)

if __name__ == '__main__':
	main()