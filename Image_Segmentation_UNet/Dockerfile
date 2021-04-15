# Docker file to run a container that will run the
# image_segmentation_unet.py in Python 3 for Tensorflow 2.4.0 (no GPU).

# Load tensorflow image for tensorflow 2.4.0 and Python 3.
FROM tensorflow/tensorflow:2.4.0

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /image-seg-unet
WORKDIR /image-seg-unet
ADD . /image-seg-unet

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.



# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Download the data for the program.
RUN curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
RUN curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
RUN tar -xf images.tar.gz
RUN tar -xf annotations.tar.gz

# Run the image_segmentation_unet.py program.
CMD ["python3", "image_segmentation_unet.py"]