FROM tensorflow/tensorflow:latest-gpu
LABEL maintainer="jhnnsrs@gmail.com"


RUN pip install csbdeep==0.7.2 arkitekt==0.4.16

# Install Kare
RUN mkdir /workspace
ADD . /workspace
WORKDIR /workspace
