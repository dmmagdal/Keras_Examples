# Docker file to run a container that will run the img_caption.py
# in Python 3 for Tensorflow 2.6.0 (no GPU).

# Load tensorflow image for tensorflow 2.6.0 and Python 3.
FROM tensorflow/tensorflow:2.6.0

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /image-captioning
WORKDIR /image-captioning
ADD . /image-captioning

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Download the data.
RUN apt-get install -y unzip wget
RUN wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
RUN wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
RUN unzip -qq Flickr8k_Dataset.zip
RUN unzip -qq Flickr8k_text.zip
RUN rm Flickr8k_Dataset.zip Flickr8k_text.zip

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Run the img_caption.py program.
CMD ["python3", "img_caption.py"]