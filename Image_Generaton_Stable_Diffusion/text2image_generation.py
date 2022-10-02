# text2image_generation.py
# Generate new images using KerasCV's StableDiffusion model.
# Source: https://keras.io/guides/keras_cv/generate_images_with_stable_
#	diffusion/
# Overview
# This example will demonstrate how to generate novel images based on a
# text prompt using the KerasCV implementation of stability.ai's
# text-to-image model, Stable Diffusion.
# Stable Diffusion is a powerful, open-source text-to-image generation
# model. While there exist multiple open-source implementations that
# allow you to easily create images from textual prompts, KerasCV's
# offer a few distinct advantages. These include XLA compilation and
# mixed precision support, which together achieve state-of-the-art
# generation speed.
# This guide will explore KerasCV's Stable Diffusion implementation,
# show how to use these powerful performance boosts, and explore the
# performance benefits they offer.
# Tensorflow 2.7
# Python 3.7
# Windows/MacOS/Linux


import time
import keras_cv
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def main():
	# First, construct a model.
	model = keras_cv.models.StableDiffusion(
		img_width=512, img_height=512
	)

	# Next, give it a prompt. Note, the arguments for
	# model.text_to_image() are:
	# -> prompt, string to encode (must be 77 tokens or shorter)
	# -> batch_size, 
	images = model.text_to_image(
		"A fighterjet flying over the desert",
		batch_size=3
	)


	def plot_images(images, name):
		plt.figure(figsize=(20, 20))
		for i in range(len(images)):
			ax = plt.subplot(1, len(images), i + 1)
			# plt.imshow(images[i])
			plt.axis("off")
			plt.savefig(f"name_{i}.png")


	plot_images(images, "initial_inference")

	# This is not all the model can do. Here are some more complex
	# prompts:
	images = model.text_to_image(
		"cute magical flying dog, fantasy art, "
		"golden color, high quality, highly detailed, elegant, sharp focus, "
		"concept art, character concepts, digital painting, mystery, adventure",
		batch_size=3,
	)
	plot_images(images, "initial_inference_complex")

	# How does it work?
	# Stable Diffusion is a kind of "latent diffusion model". You may
	# be familiar with the idea of super-resolution: it's possible to
	# train a deep learning model to denoise an input image -- and
	# thereby turn it into a higher-resolution version. The deep
	# learning model doesn't do this by magically recovering the
	# information that's missing from the noisy, low-resolution input
	# -- rather, the model uses its training data-distribution to
	# hallucinate the visual details that would be most likely given
	# the input. To learn more about super-resolution, you can check
	# out the following keras tutorials:
	# -> Image super-resolution using an efficient sub-pizel cnn
	# -> Enhanced deep residual networks for single-image 
	#	super-resolution
	# When you push this idea to the limit, you may start asking --
	# what if we just run such a model on pure noise? The model would
	# then "denoise the noise" and start hallucinating a brand new
	# image. By repeating the process multiple times, you can turn a
	# small patch of noise into an increasingly clear and
	# high-resolution artifical picture.
	# This is the key idea of latent diffusion, proposed in
	# High-resolution image synthesis with latent diffusion models in
	# 2020. To understand diffusion in depth, you can check the
	# Keras.io tutorial Denoising Diffusion Implicit Models.
	# Now, to go from latent diffusion to a text-to-image system, you
	# still need to add one key feature: the ability to control the
	# generated visual contents via prompt keywords. This is done via
	# "conditioning", a classic deep learning technique which consists
	# of concatenating to the noise patch a vector that represents a
	# bit of text, then training the model on a dataset of {image:
	# caption} pairs.
	# This gives rise to the Stable Diffusion architecture. Stable
	# Diffusion consists of three parts:
	# -> A text encoder, which turns your prompt into a latent vector.
	# -> A diffusion model, which repeatedly "denoises" a 64x64 latent
	#	image patch.
	# -> A decoder, which turns the final 64x64 latent patch into a
	#	higher-resolution 512x512 image.
	# First, your text prompts gets projected into a latent vector
	# space by the text encoder, which is simply a pretrained, frozen
	# language model. Then that prompt vector is concatenate to a
	# randomly generated noise patch, which is repeatedly "denoised" by
	# the decoder of a series of "steps" (the more steps you run the
	# clearer and nicer your image will be -- the default value is 50
	# steps).
	# Finally, the 64x64 latent image is sent through the decoder to
	# properly render it in high resolution.

	# -----------------------------------------------------------------
	# text_prompt -> [Text Encoder] -> [Diffusion Model] -> [Decoder] -> image
	#                      /\                /\          |
	#                      ||                ||          |
	#                     RNG                |_<--------_|
	# -----------------------------------------------------------------

	# All-in-all, it's a pretty simple system -- the keras
	# implementation fits in four files that represent less than 500
	# lines of code in total (in keras_cv repo, all files are under the
	# path keras_cv/keras_cv/models/generative/stable_diffusion/):
	# -> text_encoder.py: 87 LOC
	# -> diffusion_model.py: 181 LOC
	# -> text_encoder.py: 87 LOC
	# -> text_encoder.py: 87 LOC
	# But this relatively simple system starts looking like magic once
	# you train on billions of pictures and their captions. As Feynman
	# said about the universe: "It's not complicated, it's just a lot
	# of it!"

	# Perks of KerasCV
	# With several implementations of Stable Diffusion publically
	# available, why should you use keras_cv.models.StableDiffusion?
	# Aside from the easy-to-use API, KerasCV's Stable Diffusion
	# model comes with some powerful advantages, including:
	# -> Graph mode execution
	# -> XLA compilation through jit_compile=True
	# -> Support for mixed precision computation
	# When tehse are combined, the KerasCV Stable Diffusion model runs
	# orders of magnitude faster than naive implementations. This
	# section shows how to enabled all these features, and the
	# resulting performance gain yielded from using them.
	# For the purposes of comparison, we ran benchmarks comparing the
	# runtime of the HuggingFace diffusers implementation of Stable
	# Diffusion against the KerasCV implementation. Both
	# implementations were tasked to generate 3 images with a step
	# count of 50 for each image. In this benchmark, we used a Tesla T4
	# GPU.
	# All of our benchmarks are open source on GitHub, and may be
	# re-run on Colab to reproduce the results. The results from the
	# benchmark are displayed in the table below:
	# GPU 			Model 					Runtime
	# Tesla T4 		KerasCV (warm start) 	28.97s
	# Tesla T4 		diffusers (warm start) 	41.33s
	# Tesla V100 	KerasCV (warm start) 	12.45
	# Tesla V100 	diffusers (warm start) 	12.72
	# 30% improvement in execution time on the Tesla T4! While the
	# improvement is much lower on the V100, we generally expect the
	# results of the benchmark to consistently favor the KerasCV across
	# all Nvidia GPUs.
	# For the sake of completeness, both cold-start and warm-start 
	# generation times are reported. Cold-start execution time includes
	# the one-time cost of model creating and compilation, and is
	# therefore negligable in a production environment (where you would
	# reuse the same model instance many times). Regardless, here are
	# the cold-start numbers:
	# GPU 			Model 					Runtime
	# Tesla T4 		KerasCV (cold start) 	83.47s
	# Tesla T4 		diffusers (cold start) 	46.27s
	# Tesla V100 	KerasCV (cold start) 	76.43
	# Tesla V100 	diffusers (cold start) 	13.90
	# While the runtime results from running this guid may vary, in our
	# testing the KerasCV implementation of Stable Diffusion is
	# significantly faster than its PyTorch counterpart. This may be
	# largely attributed to XLA compilation.
	# NOTE: The performance of each optimization can vary significantly
	# between hardare setups.

	# To get started, let's first benchmark our unoptimized model:
	benchmark_result = []
	start = time.time()
	images = model.text_to_image(
		"A cute otter in a rainbow whirlpool holding shells, watercolor",
		batch_size=3,
	)
	end = time.time()
	benchmark_result.append(["Standard",  end - start])
	plot_images(images, "initial_inference_benchmark")

	print(f"Standard model: {(end - start):.2f} seconds")
	keras.backend.clear_session() # Clear session to preserve memory

	# Mixed precision
	# "Mixed precision" consists of performing computation using
	# float16 precision, while storing weights in the float32 format.
	# This is done to take advantage of the fact that float16
	# operations are backed by significantly faster kernels than their
	# float32 counterparts on moder Nvidia GPUs.
	# Enabling mixed precision computation in Keras (and therefore
	# keras_cv.models.StableDiffusion) is as simple as calling:
	physical_devices = tf.config.list_physical_devices("GPU")
	if len(physical_devices) != 0:
		keras.mixed_precision.set_global_policy("mixed_float16")

	# That's all. Out of the box, it just works.
	model = keras_cv.models.StableDiffusion()

	print("Compute dtype:", model.diffusion_model.compute_dtype)
	print(
		"Variable dtype:",
		model.diffusion_model.variable_dtype,
	)

	# As you can see (if you have an Nvidia GPU enabled), the model 
	# constructed above now uses mixed precision computation;
	# leveraging the speed of float16 operations for computation, while
	# storing variables in float32 precision.

	# Warm up model to run graph tracing before benchmarking.
	model.text_to_image("warming up the model", batch_size=3)

	start = time.time()
	images = model.text_to_image(
		"a cute magical flying dog, fantasy art, "
		"golden color, high quality, highly detailed, elegant, sharp focus, "
		"concept art, character concepts, digital painting, mystery, adventure",
		batch_size=3,
	)
	end = time.time()
	benchmark_result.append(["Mixed Precision", end - start])
	plot_images(images, "mixed_precision_inference")

	print(f"Mixed precision model: {(end - start):.2f} seconds")
	keras.backend.clear_session()

	# XLA compilation
	# Tensorflow comes with the XLA: Accelerated Linear Algebra
	# compiler built in. keras_cv.models.StableDiffusion supports a
	# jit_compile argument out of the box. Setting this argument to
	# True enables XLA compilation, resulting in a significant
	# speed-up.

	# Set back to the default for benchmarking purposes.
	keras.mixed_precision.set_global_policy("float32")

	model = keras_cv.models.StableDiffusion(jit_compile=True)
	# Before we benchmark the model, we run inference once to make sure
	# the TensorFlow graph has already been traced.
	images = model.text_to_image("An avocado armchair", batch_size=3)
	plot_images(images, "xla_inference")

	# Let's benchmark the XLA model.
	start = time.time()
	images = model.text_to_image(
		"A cute otter in a rainbow whirlpool holding shells, watercolor",
		batch_size=3,
	)
	end = time.time()
	benchmark_result.append(["XLA", end - start])
	plot_images(images, "mixed_precision_xla_inference")

	print(f"With XLA: {(end - start):.2f} seconds")
	keras.backend.clear_session()

	# On an A100 GPU, we get about a 2x speedup. Nice!

	# Putting it all together
	# So, how do you assemble the world's most performant stable
	# diffusion inference pipeline (as of Sept 2022)? With the
	# following lines of code:
	if len(physical_devices) != 0:
		keras.mixed_precision.set_global_policy("mixed_float16")
	model = keras_cv.models.StableDiffusion(jit_compile=True)

	# And to use it:

	# Let's make sure to warm up the model
	images = model.text_to_image(
		"Teddy bears conducting machine learning research",
		batch_size=3,
	)
	plot_images(images, "mixed_precision_xla_inference_warmup")

	# Exactly how fast is it? Let's find out:
	start = time.time()
	images = model.text_to_image(
		"A mysterious dark stranger visits the great pyramids of egypt, "
		"high quality, highly detailed, elegant, sharp focus, "
		"concept art, character concepts, digital painting",
		batch_size=3,
	)
	end = time.time()
	benchmark_result.append(["XLA + Mixed Precision", end - start])
	plot_images(images, "mixed_precision_xla_inference_benchmark")

	print(f"XLA + mixed precision: {(end - start):.2f} seconds")

	# And check out the results.
	print("{:<20} {:<20}".format("Model", "Runtime"))
	for result in benchmark_result:
		name, runtime = result
		print("{:<20} {:<20}".format(name, runtime))

	# Conclusions
	# KerasCV offers a state-of-the-art implementation of Stable
	# Diffusion -- and through the use of XLA and mixed precision, it
	# delivers the fastest Stable Diffusion pipeline available as of
	# September 2022.
	# Normally, at the end of a keras.io tutorial you are left with
	# some future directions to continue in to learn. This time we
	# leave you with one idea:
	# Go run your own prompts through the model! It is an absolute
	# blast!
	# If you have your own Nvidia GPU, or a M1 Macbook Pro, you can
	# also run the model locally on your machine. (Note that when
	# running on a M1 Macbook Pro, you should not enable mixed
	# precision, as it is not yet well supported by Apple's Metal
	# runtime).

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()