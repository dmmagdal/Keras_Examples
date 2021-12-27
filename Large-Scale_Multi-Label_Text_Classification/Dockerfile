# Docker file to run a container that will run the multi-label_class.py
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
RUN mkdir /multi-label-class
WORKDIR /multi-label-class
ADD . /multi-label-class

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Run the multi-label_class.py program.
CMD ["python3", "multi-label_class.py"]