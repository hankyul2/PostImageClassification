FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
MAINTAINER consistant1y@ajou.ac.kr

COPY requirements.txt /usr/src/app/
WORKDIR /usr/src/app
RUN apt-get update -qq && apt-get upgrade -y -qq &&\
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app
EXPOSE 8888 6006
