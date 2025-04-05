#!/bin/bash

apt-get update && apt-get install -y \
  build-essential \
  cmake \
  libboost-all-dev \
  libatlas-base-dev \
  libdlib-dev \
  libsm6 \
  libxext6 \
  libxrender-dev \
  python3-dev
