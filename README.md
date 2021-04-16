# Keras_Examples


## Source Index for all Keras Examples

https://keras.io/examples/


### DCGAN Generate Faces
Example Type: Generative Deep Learning

### DeepDream
Example Type: Generative Deep Learning

### Few-shot Learning Reptile
Example Type: Computer Vision

### Image Classification with Vision Transformer
Example Type: Computer Vision

Note: Unable to verify/run due to Out of Memory (OOM) issue on first epoch of training (on Dell Desktop).

### Image Segmentation U-Net
Example Type: Computer Vision

### Knowledge Distillation
Example Type: Computer Vision

### Neural Style Transfer
Example Type: Generative Deep Learning

### Next Frame Prediction
Example Type: Computer Vision

### Object Detection
Example Type: Computer Vision

### Q Learning Atari Breakout
Example Type: Reinforcement Learning

Note: Unable to verify/run due to issues with OpenAI baselines module.

### Text Classification with Switch Transformer
Example Type: Natural Language Processing

### Text Classification with Transformer
Example Type: Natural Language Processing

### Timeseries Anomaly Detection with Autoencoder
Example Type: Timeseries

### WGAN-GP
Example Type: Generative Deep Learning


### Notes:
Dockerfiles for GPU usage (e.g. Dockerfile-gpu) to not appear to work at the moment on Windows systems (such as Dell Desktop and Lenovo Laptop) but are rather designed for work on Linux systems with GPUs enabled (presumably). For the Linux systems, it is presumed that the NVIDIA and Docker environments are already set up to match the Dockerfile.

Machines Dell Desktop and Lenovo Laptop have Tensorflow natively installed on them. However, Dell Desktop uses Tensorflow v2.4 and is equipped with a GPU while Lenovo Laptop has Tensorflow v1.15 and is a CPU only device.