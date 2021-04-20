# Docker file to run a container that will run the cycleGAN.py in
# Python 3 for Tensorflow 2.4.0 (no GPU).

# Load tensorflow image for tensorflow 2.4.0 and Python 3.
FROM tensorflow/tensorflow:2.4.0

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /cycle-gan
WORKDIR /cycle-gan
ADD . /cycle-gan

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Download additional data.
RUN apt install unzip
RUN curl -LO https://github.com/AakashKumarNain/CycleGAN_TF2/releases/download/v1.0/saved_checkpoints.zip
RUN unzip -qq saved_checkpoints.zip

# Run the cycleGAN.py program.
CMD ["python3", "cycleGAN.py"]