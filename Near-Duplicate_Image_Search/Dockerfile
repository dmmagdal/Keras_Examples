# Docker file to run a container that will run the
# near_duplicate_imagesearch.py in Python 3 for Tensorflow
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
RUN mkdir /neardupe_imgsearch
WORKDIR /neardupe_imgsearch
ADD . /neardupe_imgsearch

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Download tools to download and extract the datasets.
RUN apt install wget
RUN apt install unzip

# Download and extract datasets. Note that this will take quite some
# time to do depending on the internet connection.
RUN wget -q https://git.io/JuMq0 -O flower_model_bit_0.96875.zip
RUN unzip -qq flower_model_bit_0.96875.zip
RUN rm flower_model_bit_0.96875.zip

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Run the near_duplicate_imagesearch.py program.
CMD ["python3", "near_duplicate_imagesearch.py"]