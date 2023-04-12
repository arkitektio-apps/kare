FROM tensorflow/tensorflow:latest-gpu
LABEL maintainer="jhnnsrs@gmail.com"


RUN pip install csbdeep==0.7.2 
RUN pip install "arkitekt[cli]==0.4.74"

# Install Kare
RUN mkdir /workspace
ADD . /workspace
WORKDIR /workspace
