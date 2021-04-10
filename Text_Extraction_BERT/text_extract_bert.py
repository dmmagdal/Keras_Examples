# text_extract_bert.py
# Fine tune pretrained BERT from HuggingFace Transformers on SQuAD.
# Introduction. This demonstration uses SQuAD (Stanford
# Question-Answering Dataset). In SQuAD, an input consists of a
# question, and a paragraph for context. The goal is to find the span
# of textin the paragraph that answers the question. Performance is
# evaluated on the data with the "Exact Match" metric, which measures
# the percentage of predictions that exactly match any one of the
# ground-truth answers.

# A BERT model is fine-tuned to perform the task as follows:
# 1) Feed the context and the question as inputs to BERT.
# 2) Take two vectors S and T with dimensions equal to that of hidden
#	states in BERT.
# 3) Compute the probability of each token being the start and end of
#	the answer span. The probability of a token being the start of the
#	answer is given by a dot product between S and the representation
#	of the token in the last layer of BERT, followed by a softmax over
#	all tokens. The probability of a token being the end of the answer
#	is computed similarly with the vector T.
# 4) Fine-tune BERT and learn S and T along the way.
# Source: https://keras.io/examples/nlp/text_extraction_with_bert/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import os
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	max_len = 384
	configuration = BertConfig()

	# Setup the BERT tokenizer. Save the slow pretrained tokenizer.
	slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	save_path = "./bert_base_uncased/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	slow_tokenizer.save_pretrained(save_path)

	# Load the fast tokenizer from saved file.
	tokenizer = BertWordPieceTokenizer(
		"./bert_base_uncased/vocab.txt", lowercase=True
	)

	# Load the data.
	train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
	train_path = keras.utils.get_file("train.json", train_data_url)
	eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
	eval_path = keras.utils.get_file("eval.json", eval_data_url)

	# Preprocess the data.
	# 1) Go through the JSON file and store every record as a
	#	SquadExample object.
	# 2) Go through each SquadExample and create x_train, y_train,
	#	x_eval, y_eval.
	with open(train_path) as f:
		raw_train_data = json.load(f)

	with open(eval_path) as f:
		raw_eval_data = json.load(f)

	train_squad_examples = create_squad_examples(
		raw_train_data, max_len, tokenizer
	)
	x_train, y_train = create_input_targets(train_squad_examples)
	print(f"{len(train_squad_examples)} training points created.")

	eval_squad_examples = create_squad_examples(
		raw_eval_data, max_len, tokenizer
	)
	x_eval, y_eval = create_input_targets(eval_squad_examples)
	print(f"{len(eval_squad_examples)} training points created.")

	# Create the Question-Answering model using BERT and Functional
	# API. The code should preferably run on a Google Colab TPU
	# runtime. With Colab TPUs, each epoch will take 5-6 minutes.
	#use_tpu = True
	use_tpu = False
	if use_tpu:
		# Create distribution strategy.
		tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
		tf.config.experimental_connect_to_cluster(tpu)
		tf.tpu.experimental.initialize_tpu_system(tpu)
		strategy = tf.distribute.experimental.TPUStrategy(tpu)\

		# Create model.
		with strategy.scope():
			model = create_model(max_len)
	else:
		model = create_model(max_len)
	model.summary()

	# Train and evaluate.
	exact_match_callback = ExactMatch(x_eval, y_eval)
	model.fit(
		x_train, 
		y_train, 
		epochs=1, # For demonstration, 3 epochs are recommended.
		verbose=2,
		batch_size=64,
		callbacks=[exact_match_callback],
	)

	# Exit the program.
	exit(0)


class SquadExample:
	def __init__(self, question, context, start_char_idx, answer_text, all_answers):
		self.question = question
		self.context = context
		self.start_char_idx = start_char_idx
		self.answer_text = answer_text
		self.all_answers = all_answers
		self.skip = False


	def preprocess(self, max_len, tokenizer):
		context = self.context
		question = self.question
		answer_text = self.answer_text
		start_char_idx = self.start_char_idx

		# Clean context, answer, and question.
		context = " ".join(str(context).split())
		question = " ".join(str(question).split())
		answer = " ".join(str(answer_text).split())

		# Find end character index of answer in context.
		end_char_idx = start_char_idx + len(answer)
		if end_char_idx >= len(context):
			self.skip = True
			return

		# Mark the character indices in context that are in answer.
		is_char_in_ans = [0] * len(context)
		for idx in range(start_char_idx, end_char_idx):
			is_char_in_ans[idx] = 1

		# Tokenize context.
		tokenized_context = tokenizer.encode(context)

		# Find tokens that were created from answer characters.
		ans_token_idx = []
		for idx, (start, end) in enumerate(tokenized_context.offsets):
			if sum(is_char_in_ans[start:end]) > 0:
				ans_token_idx.append(idx)

		if len(ans_token_idx) == 0:
			self.skip = True
			return

		# Find start and end token index for tokens from answer.
		start_token_idx = ans_token_idx[0]
		end_token_idx = ans_token_idx[-1]

		# Tokenize the question.
		tokenized_question = tokenizer.encode(question)

		# Create inputs.
		input_ids = tokenized_context.ids + tokenized_question.ids[1:]
		token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
			tokenized_question.ids[1:]
		)
		attention_mask = [1] * len(input_ids)

		# Pad and create attention masks.
		padding_length = max_len - len(input_ids)
		if padding_length > 0: # pad
			input_ids = input_ids + ([0] * padding_length)
			attention_mask = attention_mask + ([0] * padding_length)
			token_type_ids = token_type_ids + ([0] * padding_length)
		elif padding_length < 0: # skip
			self.skip = True
			return

		self.input_ids = input_ids
		self.token_type_ids = token_type_ids
		self.attention_mask = attention_mask
		self.start_token_idx = start_token_idx
		self.end_token_idx = end_token_idx
		self.context_token_to_char = tokenized_context.offsets


def create_squad_examples(raw_data, max_len, tokenizer):
	squad_examples = []
	for item in raw_data["data"]:
		for para in item["paragraphs"]:
			context = para["context"]
			for qa in para["qas"]:
				question = qa["question"]
				answer_text = qa["answers"][0]["text"]
				all_answers = [_["text"] for _ in qa["answers"]]
				start_char_idx = qa["answers"][0]["answer_start"]
				squad_eg = SquadExample(
					question, context, start_char_idx, answer_text, all_answers
				)
				squad_eg.preprocess(max_len, tokenizer)
				squad_examples.append(squad_eg)
	return squad_examples


def create_input_targets(squad_examples):
	dataset_dict = {
		"input_ids": [],
		"token_type_ids": [],
		"attention_mask": [],
		"start_token_idx": [],
		"end_token_idx": [],
	}
	for item in squad_examples:
		if item.skip == False:
			for key in dataset_dict:
				dataset_dict[key].append(getattr(item, key))
	for key in dataset_dict:
		dataset_dict[key] = np.array(dataset_dict[key])

	x = [
		dataset_dict["input_ids"],
		dataset_dict["token_type_ids"],
		dataset_dict["attention_mask"],
	]
	y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
	return x, y


def create_model(max_len):
	# BERT encoder.
	encoder = TFBertModel.from_pretrained("./bert-base-uncased")

	# QA model.
	input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
	token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
	attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
	embedding = encoder(
		input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
	)[0]

	start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
	start_logits = layers.Flatten()(start_logits)

	end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
	end_logits = layers.Flatten()(end_logits)

	start_probs = layers.Activation(keras.activations.softmax)(start_logits)
	end_probs = layers.Activation(keras.activations.softmax)(end_logits)

	model = keras.Model(
		inputs=[input_ids, token_type_ids, attention_mask],
		outputs=[start_probs, end_probs]
	)
	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
	optimizer = keras.optimizers.Adam(lr=5e-5)
	model.compile(optimizer=optimizer, loss=[loss, loss])
	return model


# Create evaluation callback. This callback will compute the exact
# match score using the validation data after every epoch.
def normalize_text(text):
	text = text.lower()

	# Remove punctuations.
	exclude = set(string.punctuation)
	text = "".join(ch for ch in text if ch not in exclude)

	# Remove articles.
	regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
	text = re.sub(regex, " ", text)

	# Remove extra white space.
	text = " ".join(text.split())
	return text


# Each SquadExample object contains the character level offsets for
# each token in its input paragraph. They will be used to get back the
# span of text corresponding to the tokens between the predicted start
# and end tokens. All the ground-truth answers are also present in each
# SquadExample object. Calculate the percentage of data points where
# the span of text obtained from model predictions matches one of the
# ground-truth answers.
class ExactMatch(keras.callbacks.Callback):
	def __init__(self, x_eval, y_eval):
		self.x_eval = x_eval
		self.y_eval = y_eval


	def on_epoch_end(self, epoch, logs=None):
		pred_start, pred_end = self.model.predict(self.x_eval)
		count = 0
		eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
		for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
			squad_eg = eval_examples_no_skip[idx]
			offsets = squad_eg.context_token_to_char
			start = np.argmax(start)
			end = np.argmax(end)
			if start >= len(offsets):
				continue
			pred_char_start = offsets[start][0]
			if end < len(offsets):
				pred_char_end = offsets[end][1]
				pred_ans = squad_eg.context[pred_char_start:pred_char_end]
			else:
				pred_ans = squad_eg.context[pred_char_start:]

			normalize_pred_ans = normalize_text(pred_ans)
			normalize_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
			if normalize_pred_ans in normalize_true_ans:
				count += 1
		acc = count / len(self.y_eval[0])
		print(f"\nepoch={epoch + 1}, exact match score={acc:.2f}")


if __name__ == '__main__':
	main()