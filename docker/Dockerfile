FROM nvidia/cuda:11.0.3-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install python
RUN \
  apt-get update && \
  apt-get install -y python python-dev python3-pip && \
  apt-get install -y git ffmpeg && \
  apt-get install -y cython python-numpy && \
  rm -rf /var/lib/apt/lists/*

# Install pytorch
RUN \
  pip3 install torch torchvision torchaudio

# Install dependencies
RUN \
  pip install docopt pyyaml numpy matplotlib tqdm Pillow tensorflow scipy

# Clone the repository
RUN \
  git clone https://github.com/sergeytulyakov/mocogan.git /mocogan

# Define working directory
WORKDIR /mocogan

# Define default command
CMD ["bash"]
