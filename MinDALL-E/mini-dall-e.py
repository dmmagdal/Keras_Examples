# min-dall-e.py
# Copy of a notebook from Kaggle that uses a smaller Dall-e model to
# generate images given a text prompt.
# Source: https://www.kaggle.com/annas82362/mindall-e
# Pytorch
# Windows/MacOS/Linux
# Python 3.7


import os
import math
import time
import clip
import torch
import numpy as np
from PIL import Image
from dalle.models import Dalle
from dalle.utils.utils import clip_score
from rudalle import get_realesrgan
from rudalle.pipelines import show, super_resolution


def main():
	# Set up cuda devices.
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# Initialize DALL-E and CLIP models.
	realesrgan = get_realesrgan("x2", device=device)
	model = Dalle.from_pretrained("minDALL-E/1.3B")
	model.to(device=device)
	model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
	model_clip.to(device=device)
	torch.cuda.empty_cache()

	# Prompt the AI.
	prompt = "A painting of avocado with top hat in the frame"

	# Number of images to be generated.
	num_candidates = 256

	# Images per batch.
	batch_size = 32

	# Number of images to be shown after re-ranking with clip
	num_show = 16

	# Set this to False to disable realesrgan.
	use_realesrgan = True

	# If nonzero, limit the sampled tokens to the top k values.
	top_k = 128

	# If nonzero, limit the sampled tokens to the cumulative
	# probability. 
	top_p = None

	# Controls the "craziness" of the generation.
	temperature = 0.7

	images = []
	print(prompt)

	for i in range(int(num_candidates / batch_size)):
		images.append(model.sampling(
			prompt=prompt,
			top_k=top_k,
			top_p=top_p,
			softmax_temperature=temperature,
			num_candidates=batch_size,
			device=device
		).cpu().numpy())
		torch.cuda.empty_cache()

	images = np.concatenate(images)
	images = np.transpose(images, (0, 2, 3, 1))

	if num_candidates > 1:
		rank = clip_score(
			prompt=prompt,
			images=images,
			model_clip=model_clip,
			preprocess_clip=preprocess_clip,
			device=device
		)
		torch.cuda.empty_cache()
		images = images[rank]

	num_show = num_show if num_candidates >= num_show else num_candidates

	images = [
		Image.fromarray((images[i] * 255).astype("uint8")) 
		for i in range(num_show)
	]

	if use_realesrgan:
		images = super_resolution(images, realesrgan)

	print(prompt)
	show(
		images, 
		int(math.ceil(num_show / math.sqrt(num_show)))
	) # Perfect square when using square numbers by litevex#6982
	for i in range(len(images)):
		#display(images[i])
		images[i].save(f"../{i}.png")

	# Can download the image by looking at /kaggle/working click 3 dots
	# and download.
	time.sleep(120000)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()