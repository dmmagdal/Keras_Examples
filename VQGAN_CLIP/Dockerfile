# Docker file to run a container that will run the vqganclip.py in
# Python 3 for Pytorch (no GPU).

# Load tensorflow image for Pytorch and Python 3.
FROM pytorch/pytorch:latest

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /vqgan
WORKDIR /vqgan
ADD . /vqgan

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Some extra environment setup.
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git curl

# Download all models. Alternate download URLs listed here:
# https://www.reddit.com/r/bigsleep/comments/p96pnh/a_site_that_hosts_some_vqgan_models_isnt_working/
#ImageNet 1024 (Error: Could not resolve host mirror.io.community)
#RUN curl -L -o vqgan_imagenet_f16_1024.yaml -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml'
#RUN !curl -L -o vqgan_imagenet_f16_1024.ckpt -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt'

#ImageNet 16384
RUN curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
RUN curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'

#OpenImages 8192
RUN curl -L -o vqgan_openimages_f16_8192.yaml -C - 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
RUN curl -L -o vqgan_openimages_f16_8192.ckpt -C - 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1'

#COCO
#RUN curl -L -o coco.yaml -C - 'https://dl.nmkd.de/ai/clip/coco/coco.yaml'
#RUN curl -L -o coco.ckpt -C - 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt'

#FacesHQ (Takes too long to download)
#RUN curl -L -o faceshq.yaml -C - 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT'
#RUN curl -L -o faceshq.ckpt -C - 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt'

#WikiArt 1024 (Error: Could not resolve host mirror.io.community)
#RUN curl -L -o wikiart_1024.yaml -C - 'http://mirror.io.community/blob/vqgan/wikiart.yaml'
#RUN curl -L -o wikiart_1024.ckpt -C - 'http://mirror.io.community/blob/vqgan/wikiart.ckpt'

#WikiArt 16384 (Error: Could not resolve host mirror.io.community)
#RUN curl -L -o wikiart_16384.yaml -C - 'http://mirror.io.cozzzzmmunity/blob/vqgan/wikiart_16384.yaml' 
#RUN curl -L -o wikiart_16384.ckpt -C - 'http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt'

#S-FLCKR
#RUN curl -L -o sflckr.yaml -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1'
#RUN curl -L -o sflckr.ckpt -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1'

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -q git+https://github.com/openai/CLIP.git
RUN git clone https://github.com/CompVis/taming-transformers.git
RUN mkdir steps
RUN cp vqganclip.py ./taming-transformers
RUN cp *.yaml ./taming-transformers
RUN cp *.ckpt ./taming-transformers

# Run the vqganclip.py program.
CMD ["python3", "./taming-transformers/vqganclip.py"]