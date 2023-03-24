FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN apt-get install -y pip
RUN pip install pyaudio