FROM tensorflow/tensorflow:latest-gpu
LABEL maintainer="jhnnsrs@gmail.com"


RUN pip install csbdeep==0.7.2 
RUN pip install "arkitekt[all]==0.5.59"
RUN pip install "pydantic<2"

# Install Kare
RUN mkdir /workspace
ADD . /workspace
WORKDIR /workspace
