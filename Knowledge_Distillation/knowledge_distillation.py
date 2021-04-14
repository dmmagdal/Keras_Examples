# knowledge_distillation.py
# Implementation of classical knowledged distillation.
# Knowledge distillation is a procedure for model compression, in which
# a small (student) model is trained to match a large pre-trained
# (teacher) model. Knowledge is transferred from the teacher model to
# the student by minimizing a loss function, aimed at matching softened
# teacher logits as well as ground-truth labels. The logits are
# softened by applying a "temperature" scaling function in the softmax,
# effectively smoothing out the probability distribution and revealing
# inter-class relationships learned by the teacher.
# Source: https://keras.io/examples/vision/knowledge_distillation/
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	# Create student and teacher models. Initially, we create a teacher
	# model and a smaller student model. Both models are convolutional
	# neural networks and created using Sequential(), but could be any
	# keras model.
	# Create the teacher.
	teacher = keras.Sequential(
		[
			keras.Input(shape=(28, 28, 1)),
			layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
			layers.LeakyReLU(alpha=0.2),
			layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
			layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
			layers.Flatten(),
			layers.Dense(10),
		],
		name="teacher",
	)

	# Create the student.
	student = keras.Sequential(
		[
			keras.Input(shape=(28, 28, 1)),
			layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
			layers.LeakyReLU(alpha=0.2),
			layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
			layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
			layers.Flatten(),
			layers.Dense(10),
		],
		name="student",
	)

	# Clone student for later comparison.
	student_scratch = keras.models.clone_model(student)

	# Prepare the dataset. The dataset used for training the teacher
	# and distilling the teacher is MNIST, and the procedure would be
	# equivalent for any other dataset, e.g. CIFAR-10, with a suitable
	# choice of models. Both the student and teacher are trained on the
	# training set and evaluated on the test set.
	# Prepare the train and test dataset.
	batch_size = 64
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

	# Normalize data.
	x_train = x_train.astype("float32") / 255.0
	x_train = np.reshape(x_train, (-1, 28, 28, 1))

	x_test = x_test.astype("float32") / 255.0
	x_test = np.reshape(x_test, (-1, 28, 28, 1))

	# Train the teacher. In knowledge distillation we assume that the
	# teacher is trained and fixed. Thus, we start by training the
	# teacher model on the training dataset in the usual way.
	# Train teacher as usual.
	teacher.compile(
		optimizer=keras.optimizers.Adam(),
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[keras.metrics.SparseCategoricalAccuracy()],
	)

	# Train and evaluate teacher on data.
	teacher.fit(x_train, y_train, epochs=5)
	teacher.evaluate(x_test, y_test)

	# Distill teacher to student. We have already trained the teacher
	# model, and we only need to initialize a Distiller(student,
	# teacher) instance, compile() it with the desired losses,
	# hyperparameters and optimizer, and distill the teacher to the
	# student.
	# Initialize and compile distiller.
	distiller = Distiller(student=student, teacher=teacher)
	distiller.compile(
		optimizer=keras.optimizers.Adam(),
		metrics=[keras.metrics.SparseCategoricalAccuracy()],
		student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		distillation_loss_fn=keras.losses.KLDivergence(),
		alpha=0.1,
		temperature=10,
	)

	# Distill teacher to student.
	distiller.fit(x_train, y_train, epochs=3)

	# Evaluate student on test dataset.
	distiller.evaluate(x_test, y_test)

	# Train student from scratch for comparison.
	student_scratch.compile(
		optimizer=keras.optimizers.Adam(),
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[keras.metrics.SparseCategoricalAccuracy()],
	)

	# Train and evaluate student trained from scratch.
	student_scratch.fit(x_train, y_train, epochs=3)
	student_scratch.evaluate(x_test, y_test)

	# If the teacher trained for 5 full epochs and the student is
	# distilled on this teacher for 3 full epochs, you should in this
	# example experience a performance boost compared to training the
	# same student model from scratch, and even compared to the teacher
	# itself. You should expect the teacher to have an accuracy around
	# 97.6% the student trained from scratch should be around 97.6%,
	# and the distilled student should be around 98.1%. Remove or try
	# out different seeds to use different weight initializations.

	# Exit the program.
	exit(0)


# Construct Distiller() class. The custom Distiller() class, overrides
# the Model methods train_step, test_step, and compile(). In order to
# use the distiller, we need:
# 1) A trained teacher model.
# 2) A student model to train.
# 3) A student loss function on the difference between student#
#	predictions and ground-truth.
# 4) A distillation loss function, along with a temperature, on the
#	difference between the soft student predictions and the soft
#	teacher labels.
# 5) An alpha factor to weight the student and distillation loss.
# 6) An optimizer for the student and (optional) metrics to evaluate
#	performance.
# In the train_step method, we perform a forward pass of both the
# teacher and student, calculate the loss with weighting of the
# student_loss and distillation_loss by alpha and 1 - alpha,
# respectively, and perform the backward pass. Note: only the student
# weights are updated, and therefore, we only calculate the gradients
# for the student weights.
# In the test_step method, we evaluate the student model on the
# provided dataset.
class Distiller(keras.Model):
	def __init__(self, student, teacher):
		super(Distiller, self).__init__()
		self.teacher = teacher
		self.student = student


	# Compile the distller.
	# @param: optimizer, keras optimizer for the student weights.
	# @param: metrics, keras metrics for evaluation.
	# @param: student_loss_fn, loss function of difference between
	#	student predictions and ground-truth.
	# @param: distillation_loss_fn, loss function of difference between
	#	soft student predictions and soft teacher predictions.
	# @param: alpha, weight to student_loss_fn and 1 - alpha to
	#	distillation_loss_fn.
	# @param: temperature, temperature for softening probability
	#	distributions. Larger temperature gives softer distributions.
	# @return: returns nothing.
	def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, 
			alpha=0.1, temperature=3):
		super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
		self.student_loss_fn = student_loss_fn
		self.distillation_loss_fn = distillation_loss_fn
		self.alpha = alpha
		self.temperature = temperature


	def train_step(self, data):
		# Unpack data.
		x, y = data

		# Forward pass of teacher.
		teacher_predictions = self.teacher(x, training=False)

		with tf.GradientTape() as tape:
			# Forward pass of student.
			student_predictions = self.student(x, training=True)

			# Compute losses.
			student_loss = self.student_loss_fn(y, student_predictions)
			distillation_loss = self.distillation_loss_fn(
				tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
				tf.nn.softmax(student_predictions / self.temperature, axis=1),
			)
			loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

		# Compute gradients.
		trainable_vars = self.student.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights.
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update the metrics configured in compile().
		self.compiled_metrics.update_state(y, student_predictions)

		# Return a dict of performance.
		results = {m.name: m.result() for m in self.metrics}
		results.update(
			{"student_loss": student_loss, "distillation_loss": distillation_loss}
		)
		return results


	def test_step(self, data):
		# Unpack data.
		x, y = data

		# Compute predictions.
		y_prediction = self.student(x, training=False)

		# Calculate the loss.
		student_loss = self.student_loss_fn(y, y_prediction)

		# Update the metrics.
		self.compiled_metrics.update_state(y, y_prediction)

		# Return a dict of performance.
		results = {m.name: m.result() for m in self.metrics}
		results.update({"student_loss": student_loss})
		return results



if __name__ == '__main__':
	main()