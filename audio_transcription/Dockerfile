# Determined Image
FROM determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-gpu-0.19.4

WORKDIR /app
COPY ./* /app/
RUN apt-get update && \
   DEBIAN_FRONTEND="noninteractive" apt-get -y install sox unzip libsndfile1 ffmpeg
# Custom Configuration
RUN pip install wget unidecode sox gradio pytube librosa soundfile && \
   pip install Cython && \
   pip install nemo-toolkit['all']

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_PORT=8983
ENV GRADIO_SERVER_NAME="0.0.0.0"