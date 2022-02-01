# Docker file to run a container that will run the
# mini-dall-e.py in Python 3 for Pytorch (no GPU).

# Load tensorflow image for Pytorch and Python 3.
FROM pytorch/pytorch:latest

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /dalle
WORKDIR /dalle
ADD . /dalle

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Some extra environment setup.
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git
RUN echo "10" | apt-get install -y python3-opencv
#RUN apt-get install -y libgl1
#RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN git clone -q https://github.com/kakaobrain/minDALL-E.git

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

RUN cp mini-dall-e.py ./minDALL-E
RUN pip3 install -q git+https://github.com/openai/CLIP.git

# Run the mini-dall-e.py program.
CMD ["python3", "./minDALL-E/mini-dall-e.py"]