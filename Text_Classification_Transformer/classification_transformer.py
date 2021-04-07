# classifcation_transformer.py
# Simple text classifcation with a transformer.
# Source: https://keras.io/examples/nlp/text_classification_with_
# transformer/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
	# Download and prepare dataset.
	vocab_size = 20000 # Only consider the top 20k words.
	maxlen = 200 # Only consider the first 200 words of each movie
	# review.
	(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
	print(len(x_train), "Training sequences")
	print(len(x_val), "Validation sequences")
	x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
	x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

	# Create classifier model using transformer layer.
	embed_dim = 32 # Embedding size for each token.
	num_heads = 2 # Number of attention heads.
	ff_dim = 32 # Hidden layer size in feedforward network.
	dropout_rate = 0.1 # Dropout rate.
	batch_size = 32 # Batch size.
	epochs = 2 # Number of epochs.

	inputs = layers.Input(shape=(maxlen,))
	embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
	x = embedding_layer(inputs)
	transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
	x = transformer_block(x)
	x = layers.GlobalAveragePooling1D()(x)
	x = layers.Dropout(dropout_rate)(x)
	x = layers.Dense(ff_dim, activation="relu")(x)
	x = layers.Dropout(dropout_rate)(x)
	outputs = layers.Dense(2, activation="softmax")(x)

	model = keras.Model(inputs=inputs, outputs=outputs)

	# Train and evaluate.
	model.compile("adam", "sparse_categorical_crossentropy", 
					metrics=["accuracy"]
	)
	print(model.summary())
	history = model.fit(
		x_train, y_train, batch_size=batch_size, epochs=epochs,
		validation_data=(x_val, y_val)
	)

	# Exit the program.
	exit(0)


# Implement a transformer block as a layer.
class TransformerBlock(layers.Layer):
	def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
		super(TransformerBlock, self).__init__()
		self.att = layers.MultiHeadAttention(num_heads=num_heads,
											key_dim=embed_dim)
		self.ffn = keras.Sequential(
			[layers.Dense(ff_dim, activation="relu"),
			layers.Dense(embed_dim),]
		)
		self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = layers.Dropout(dropout_rate)
		self.dropout2 = layers.Dropout(dropout_rate)


	def call(self, inputs, training):
		attn_output = self.att(inputs, inputs)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(inputs + attn_output)
		ffn_output = self.ffn(out1)
		ffn_output = self.dropout2(ffn_output, training=training)
		return self.layernorm2(out1 + ffn_output)


# Implement embedding layer.
class TokenAndPositionEmbedding(layers.Layer):
	def __init__(self, maxlen, vocab_size, embed_dim):
		super(TokenAndPositionEmbedding, self).__init__()
		self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
		self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)


	def call(self, x):
		maxlen = tf.shape(x)[-1]
		positions = tf.range(start=0, limit=maxlen, delta=1)
		positions = self.pos_emb(positions)
		x = self.token_emb(x)
		return x + positions


if __name__ == '__main__':
	main()