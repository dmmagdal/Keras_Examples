# Docker file to run a container that will run the
# natural_language_imagesearch.py in Python 3 for Tensorflow
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
RUN mkdir /nl_imgsearch
WORKDIR /nl_imgsearch
ADD . /nl_imgsearch

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Download tools to download and extract the datasets.
RUN apt install wget
RUN apt install unzip

# Setup paths for downloaded datasets.
RUN mkdir datasets
#RUN mkdir datasets/annotations
#RUN mkdir datasets/train2014

# Download and extract datasets. Note that this will take quite some
# time to do depending on the internet connection.
RUN wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
RUN unzip annotations_trainval2014.zip -d datasets
RUN rm annotations_trainval2014.zip
RUN wget http://images.cocodataset.org/zips/train2014.zip
RUN unzip train2014.zip -d datasets
RUN rm train2014.zip

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Run the natural_language_imagesearch.py program.
CMD ["python3", "natural_language_imagesearch.py"]