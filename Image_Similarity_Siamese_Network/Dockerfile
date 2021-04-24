# Docker file to run a container that will run the
# image_similarity_siamese_networks.py in Python 3 for Tensorflow
# 2.4.0 (no GPU).

# Load tensorflow image for tensorflow 2.4.0 and Python 3.
FROM tensorflow/tensorflow:2.4.0

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /image-similarity
WORKDIR /image-similarity
ADD . /image-similarity

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Download all the data.
RUN pip3 install gdown
RUN gdown --id 1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34
RUN gdown --id 1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW
RUN apt install unzip
#RUN cache_dir="~/.keras"
#RUN unzip -oq left.zip -d $cache_dir
#RUN unzip -oq right.zip -d $cache_dir
RUN mkdir /image-similarity/keras
RUN unzip -oq left.zip -d "/image-similarity/keras"
RUN unzip -oq right.zip -d "/image-similarity/keras"

# Run the image_similarity_siamese_networks.py program.
CMD ["python3", "image_similarity_siamese_networks.py"]