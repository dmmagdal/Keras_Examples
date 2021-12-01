# graph_att_network.py
# An implementation of Graph Attention Networks (GATs) for node
# classification.
# Introduction
# Graph neural networks is the prefered nerual network architecture for
# processing data structured as graphs (for example, social networks or
# molecule structures), yielding better results than fully-connected
# neural networks or convolutional networks.
# This tutorial implements a specific graph neural network known as a
# Graph Attention Network (GAT) to predict labels of specific papers
# based on the papers they cite (using the Cora dataset).
# References
# For more information on GAT, see the original paper Graph Attention
# Networks as well as DGL's Graph Attention Networks documentation.
# Source: https://keras.io/examples/graph/gat_node_classification/
# Tensorflow 2.4.0
# Python 3.7
# Windows/MacOS/Linux


import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
	warnings.filterwarnings("ignore")
	pd.set_option("display.max_columns", 6)
	pd.set_option("display.max_rows", 6)
	np.random.seed(2)

	# Obtain the dataset
	# The preparation of the Cora dataset follows that of the Node 
	# classification with Graph Neural Networks tutorial. Refer to this
	# tutorial for more details on the dataset and exploratory data
	# analysis. In brief, the Cora dataset consists of two files:
	# cora.cites which contains directed links (citations) between
	# papers; and cora.content which contains features of the
	# corresponding papers and one of seven labels (the subject of the
	# paper).
	zip_file = keras.utils.get_file(
		fname="cora.tgz",
		origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
		extract=True,
	)

	data_dir = os.path.join(os.path.dirname(zip_file), "cora")

	citations = pd.read_csv(
		os.path.join(data_dir, "cora.cites"),
		sep="\t",
		header=None,
		names=["target", "source"],
	)

	papers = pd.read_csv(
		os.path.join(data_dir, "cora.content"),
		sep="\t",
		header=None,
		names=["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"],
	)

	class_values = sorted(papers["subject"].unique())
	class_idx = {name: id for id, name in enumerate(class_values)}
	paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

	papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
	citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
	citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
	papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

	print(citations)

	print(papers)

	# Split the dataset
	# Obtain random indices.
	random_indices = np.random.permutation(range(papers.shape[0]))

	# 50/50 split.
	train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
	test_data = papers.iloc[random_indices[len(random_indices) // 2 :]]

	# Prepare the graph data
	# Obtain paper indices which will be used to gather node states
	# from the graph later on when training the model.
	train_indices = train_data["paper_id"].to_numpy()
	test_indices = test_data["paper_id"].to_numpy()

	# Obtain ground truth labels corresponding to each paper_id.
	train_labels = train_data["subject"].to_numpy()
	test_labels = test_data["subject"].to_numpy()

	# Define graph, namely an edge tensor and a node feature tensor.
	edges = tf.convert_to_tensor(citations[["source", "target"]])
	node_features = tf.convert_to_tensor(papers.sort_values("paper_id").iloc[:, 1:-1])

	# Print shapes of the graph.
	print("Edges shape:\t\t", edges.shape)
	print("Node features shape:", node_features.shape)

	# Build the model
	# GAT takes as input a graph (namely an edge tensor and a node
	# feature tensor) and outputs [updated] node states. The node
	# states are, for each source node, neighborhood aggregated
	# information of N-hops (where N is decided by the number of layers
	# of the GAT). Importantly, in contrast to the graph convolutional
	# network (GCN) the GAT makes used of attention mechanisms to 
	# aggregate information from neighboring nodes. In other words,
	# instead of simply averaging/summing node states from neighbors to
	# the source node, GAT first applies normalized attention scores to
	# each neighbor node state and then sums.
	# (Multi-head) graph attention layer
	# The GAT model implements multi-head graph attention layers. The
	# MultiHeadGraphAttention layer is simply a concatenation (or
	# averaging) of multiple graph attention layers (GraphAttention),
	# each with separate learnable weights w. The GraphAttention layer
	# does the following:
	# Consider inputs node states h^{l} which are linearly transformed
	# by w^{l}, resulting in z{l}.
	# For each source node:
	# 1) Computes pair-wise attention scores 
	#	a^{l}^{T}{z^{l}+{i}||z^{l}_{j}} for all j, resulting in e_{ij}
	#	(for all j). || denotes concatenation, _{i} corresponds to the
	#	source node, and _{j} corresponds to a given 1-hop neighbor
	#	node.
	# 2) Normalizes e_{ij} via softmax, so as the sum of incoming
	#	edges' attention scores to the source node 
	#	(sum_{k}{e_{norm}_{ik}}) will add up to 1. (Notice, in this
	#	tutorial, incoming edges are defined as edges pointing from the
	#	source paper (the paper citing) to the target paper (the paper
	#	cited), which is counter intuitve. However, this seems to work
	#	better in practice. In other words, we want to learn the label
	#	of the source paper based on what it cites (target papers)).
	# 3) Applies attention scores e_{norm}_{ij} to z_{j} and adds it to
	#	the new source node state h^{l+1}_{i}, for all j.
	class GraphAttention(layers.Layer):
		def __init__(self, units, kernel_initializer="glorot_uniform",
				kernel_regularizer=None, **kwargs):
			super().__init__(**kwargs)
			self.units = units
			self.kernel_initializer = keras.initializers.get(kernel_initializer)
			self.kernel_regularizer= keras.regularizers.get(kernel_regularizer)


		def build(self, input_shape):
			self.kernel = self.add_weight(
				shape=(input_shape[0][-1], self.units),
				trainable=True,
				initializer=self.kernel_initializer,
				regularizer=self.kernel_regularizer,
			)
			self.kernel_attention = self.add_weight(
				shape=(self.units * 2, 1),
				trainable=True,
				initializer=self.kernel_initializer,
				regularizer=self.kernel_regularizer,
			)
			self.built = True


		def call(self, inputs):
			node_features, edges = inputs

			# Linearly transform node features (node states).
			node_features_transformed = tf.matmul(node_features, self.kernel)

			# (1) Compute pair-wise attention scores.
			node_features_expanded = tf.gather(node_features_transformed, edges)
			node_features_expanded = tf.reshape(
				node_features_expanded, (tf.shape(edges)[0], -1)
			)
			attention_scores = tf.nn.leaky_relu(
				tf.matmul(node_features_expanded, self.kernel_attention)
			)
			attention_scores = tf.squeeze(attention_scores, -1)

			# (2) Normalize attention scores.
			attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
			attention_scores_sum = tf.math.unsorted_segment_sum(
				data=attention_scores,
				segment_ids=edges[:, 0],
				num_segments=tf.reduce_max(edges[:, 0]) + 1,
			)
			attention_scores_sum = tf.repeat(
				attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
			)
			attention_scores_norm = attention_scores / attention_scores_sum

			# (3) Gather node states of neighbors, apply attention
			# scores and aggregate.
			node_features_neighbors = tf.gather(node_features_transformed, edges[:, 1])
			out = tf.math.unsorted_segment_sum(
				data=node_features_neighbors * attention_scores_norm[:, tf.newaxis],
				segment_ids=edges[:, 0],
				num_segments=tf.shape(node_features)[0],
			)
			return out


	class MultiHeadGraphAttention(layers.Layer):
		def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
			super().__init__(**kwargs)
			self.num_heads = num_heads
			self.merge_type = merge_type
			self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]


		def call(self, inputs):
			atom_features, pair_indices = inputs

			# Obtain outputs from each attention head.
			outputs = [
				attention_layer([atom_features, pair_indices])
				for attention_layer in self.attention_layers
			]

			# Concatenate or average the nodes states from each head.
			if self.merge_type == "concat":
				outputs = tf.concat(outputs, axis=-1)
			else:
				outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

			# Activate and return node states.
			return tf.nn.relu(outputs)


	# Implement training logic with custome train_step, test_step, and
	# predict_step methods
	# Notice, the GAT model operates on the entire graph (namely,
	# node_features and edges) in all phases (training, validation, and
	# testing). Hence, node_features and edges are passed to the
	# constructor of the keras.Model and used as attributes. The
	# difference between the phases are the indices (and labels), which
	# gathers certain output units (tf.gather(outputs, indices)).
	class GraphAttentionNetwork(keras.Model):
		def __init__(self, node_features, edges, hidden_units, num_heads,
				num_layers, output_dim, **kwargs):
			super().__init__(**kwargs)
			self.node_features = node_features
			self.edges = edges
			self.preprocess = layers.Dense(
				hidden_units * num_heads, activation="relu"
			)
			self.attention_layers = [
				MultiHeadGraphAttention(hidden_units, num_heads)
				for _ in range(num_layers)
			]
			self.output_layer = layers.Dense(output_dim)


		def call(self, inputs):
			node_features, edges = inputs
			x = self.preprocess(node_features)
			for attention_layer in self.attention_layers:
				x = attention_layer([x, edges]) + x
			outputs = self.output_layer(x)
			return outputs


		def train_step(self, data):
			indices, labels = data

			with tf.GradientTape() as tape:
				# Forward pass.
				outputs = self([self.node_features, self.edges])

				# Compute loss.
				loss = self.compiled_loss(labels, tf.gather(outputs, indices))

			# Compute gradients.
			grads = tape.gradient(loss, self.trainable_weights)

			# Apply gradients (update weights).
			optimizer.apply_gradients(zip(grads, self.trainable_weights))

			# Update metrics.
			self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

			return {m.name: m.result() for m in self.metrics}


		def predict_step(self, data):
			indices = data

			# Forward pass.
			outputs = self([self.node_features, self.edges])

			# Compute probabilities.
			return tf.nn.softmax(tf.gather(outputs, indices))


		def test_step(self, data):
			indices, labels = data

			# Forward pass.
			outputs = self([self.node_features, self.edges])

			# Compute loss.
			loss = self.compiled_loss(labels, tf.gather(outputs, indices))

			# Update metrics.
			self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

			return {m.name: m.result() for m in self.metrics}
			

	# Train and evaluate
	# Define hyper-parameters
	HIDDEN_UNITS = 100
	NUM_HEADS = 8
	NUM_LAYERS = 3
	OUTPUT_DIM = len(class_values)

	NUM_EPOCHS = 100
	BATCH_SIZE = 256
	VALIDATION_SPLIT = 0.1
	LEARNING_RATE = 3e-1
	MOMENTUM = 0.9

	loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
	accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
	early_stopping = keras.callbacks.EarlyStopping(
		monitor="val_acc", min_delta=1e-5, patience=5,
		restore_best_weights=True
	)

	# Build model.
	get_model = GraphAttentionNetwork(
		node_features, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS,
		OUTPUT_DIM
	)

	# Compile model.
	get_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

	get_model.fit(
		x=train_indices, y=train_labels, validation_split=VALIDATION_SPLIT,
		batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[early_stopping],
		verbose=2,
	)

	_, test_accuracy = get_model.evaluate(
		x=test_indices, y=test_labels, verbose=0
	)

	print("--" * 38 + f"\nTest Accuracy: {test_accuracy * 100:.1f}%")

	# Predict (probabilities)
	test_probs = get_model.predict(x=test_indices)

	mapping = {v: k for (k, v) in class_idx.items()}

	for i, (probs, label) in enumerate(zip(test_probs[:10], test_labels[:10])):
		print(f"Example {i+1}: {mapping[label]}")
		for j, c in zip(probs, class_idx.keys()):
			print(f"\tProbability of {c: <24} = {j*100:7.3f}%")
		print("---" * 20)

	# Conclusion
	# The resutls look OK! The GAT model seems to correctly predict the
	# subjects of the papers, based on what they cite, about 80-85% of
	# the time. Further improvements could be made by fine-tuning the
	# hyper-parameters of the GAT. For instance, try changing the
	# number of layers, the number of hiddent units, or the optimizer/
	# learning rate; add regularization (e.g. dropout); or modify the
	# preprocessing step. Could also try to implement self-loops (i.e.
	# paper X cites paper X) and/or make the graph undirected.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()