# Keras_Examples


## Source Index for all Keras Examples

https://keras.io/examples/


### Actor Critic Learning
Example Type: Reinforcement Learning

### Audio Speech Recognition Transformer
Example Type: Audio Data

Note: Unable to verify/run on Docker due to issue with code. Code has only been able to be confirmed working on "bare metal" TF 2.7.0 environment.

### Automatic Speech Recognition using CTC
Example Type: Audio Data

### Classification Neural Decision Forests
Example Type: Structured Data

### Compact Convolutional Transformer
Example Type: Computer Vision

Note: Unable to verify/run on Docker due to issue with code. Code has only been able to be confirmed working on "bare metal" TF 2.7.0 environment.

### ConditionalGAN
Example Type: Generative Deep Learning

### CycleGAN
Example Type: Generative Deep Learning

Note: Unable to verify/run due to issue with applying a function to dataset that requires multiple arguments.

### DCGAN Generate Faces
Example Type: Generative Deep Learning

### DeepDream
Example Type: Generative Deep Learning

### Deep Deterministic Policy Gradient
Example Type: Reinforcement Learning

### Few-shot Learning Reptile
Example Type: Computer Vision

### GauGAN Conditional Image Generation
Example Type: Generative Deep Learning

### Graph Attention Network Node Classification 
Example Type: Graph Data

### Graph Representation Learning Node2vec
Example Type: Graph Data

### Image Captioning
Example Type: Computer Vision

### Image Classification EfficientNet Finetunning
Example Type: Computer Vision

Note: This example is unfinished due to loss of interest. It has yet to be completed in such a way that it can be verified on Docker. May come back to this later.

### Image Classification with Vision Transformer
Example Type: Computer Vision

Note: Unable to verify/run due to Out of Memory (OOM) issue on first epoch of training (on Dell Desktop).

### Image Segmentation U-Net
Example Type: Computer Vision

### Image Similarity Siamese Network
Example Type: Computer Vision

### Image Super-Resolution with ESRGAN
Example Type: Computer Vision

### Image Super-Resolution Efficient Sub-Pixel CNN
Example Type: Computer Vision

### Imbalanced Classification
Example Type: Structured Data

### Keras Tuner
Example Type: N/A

### Knowledge Distillation
Example Type: Computer Vision

### Large Scale Multi Label Classification
Example Type: Natural Language Processing

### Lego Minifigure GAN
Example Type: Generative Deep Learning (Kaggle)

### Masked Language Modeling BERT
Example Type: Natural Language Processing

### MelGAN Spectrogram Inversion
Example Type: Audio Data

Note: Unable to verify/run due to issues with code. Tensorflow gives errors when training despite code being almost exact from example. Code from example page (linked notebook/github) also produces errors and fails as well.

### Mobile Vision Transformers
Example Type: Computer Vision

Note: Unable to verify/run on Docker due to issue with code. Code has only been able to be confirmed working on "bare metal" TF 2.7.0 environment.

### MultiModal Entailment
Example Type: Natural Language Processing

### Natural Language Image Search Dual Encoder
Example Type: Natural Language Processing

### Neural Style Transfer
Example Type: Generative Deep Learning

### Next Frame Prediction
Example Type: Computer Vision

### Node Classification Graph Neural Networks
Example Type: Graph Data

### Object Detection
Example Type: Computer Vision

### PixelCNN
Example Type: Generative Deep Learning

### Q Learning Atari Breakout
Example Type: Reinforcement Learning

Note: Unable to verify/run due to issues with OpenAI baselines module.

### Recommendation Transformer
Example Type: Structured Data

Note: Unable to verify/run due to issues with code. Tensorflow gives errors when training despite code being almost exact from example.

### Recommending Movies Retrieval
Example Type: Structured Data

### Text Classification with Switch Transformer
Example Type: Natural Language Processing

### Text Classification with Transformer
Example Type: Natural Language Processing

### Text Extraction BERT
Example Type: Natural Language Processing

Note: Unable to verify/run due to Out of Memory (OOM) issue (on Dell Desktop).

### Text Generation FNet
Example Type: Natural Language Processing

### Text Generation Mini GPT
Example Type: Natural Language Processing

### Timeseries Anomaly Detection with Autoencoder
Example Type: Timeseries

### Variational AutoEncoder
Example Type: Generative Deep Learning

Note: Unable to verify/run due to issues with code. Tensorflow gives errors when training despite code being almost exact from example.

### Vector Quantized Variational AutoEncoder
Example Type: Generative

Note: Unable to verify/run on Docker due to issue with code. Code has only been able to be confirmed running (not working) on "bare metal" TF 2.7.0 environment. Original code seems to also yield the same errors on Google Colab as are being seen currently on "bare metal" environment. Issue is with code in Sampling Codebook section. May update later if example author addresses/updates source.

### WGAN-GP
Example Type: Generative Deep Learning

### Wide Deep Cross Networks
Example Type: Structured Data

### Notes:
Dockerfiles for GPU usage (e.g. Dockerfile-gpu) to not appear to work at the moment on Windows systems (such as Dell Desktop and Lenovo Laptop) but are rather designed for work on Linux systems with GPUs enabled (presumably). For the Linux systems, it is presumed that the NVIDIA and Docker environments are already set up to match the Dockerfile.

-

Machines Dell Desktop and Lenovo Laptop have Tensorflow natively installed on them. However, Dell Desktop uses Tensorflow v2.4 and is equipped with a GPU while Lenovo Laptop has Tensorflow v2.4 and is a CPU only device.

-

On Windows machines (especially those running Windows 10 Home), Docker does not automatically release storage after removing/pruning containers on the hard drive. This is because the virtual hard disk for Docker does not release that memory back to the system for some reason on Windows. To reclaim that storage, do the following:

1) Run "docker system prune" in the command line. This will remove the excess container and images not in use in the docker virtual hard disk.

2) Then open a command line (Windows Terminal, Command Prompt, or Powershell) in admin mode.

3) Run the command "wsl.exe --shutdown" command to shut down WSL2 on the machine. This will cause Docker to shutdown as well.

4) Navigate to the the following path to locate the docker virtual hard disk.

Path: "C:\Users\comp_user\Appdata\Local\Docker\wsl\data"

The name of the file of the virtual hard disk usually looks like "ext4.vhdx".

5) Issuing the command "optimize-vhd -Path C:\Users\comp_user\Appdata\Local\Docker\wsl\data\ext4.vhdx Mode -full" will shrink that virtual hard disk (only works with Windows Pro or Enterprise editions).

6) On Windows 10 Home, run the following command to shrink the virtual hard disk:

CMD>diskpart

DISKPART>Select vdisk file="C:\Users\comp_user\AppData\Local\Docker\wsl\data\ext4.vhdx"

DISKPART>attach vdisk readonly

DISKPART>compact vdisk

DISKPART>detach vdisk

WARNING: This will remove all compiled images and containers in Docker. Make sure you do these steps if that is acceptable to you.

-

## Source Index for Tensorflow Examples

https://www.tensorflow.org/tutorials

