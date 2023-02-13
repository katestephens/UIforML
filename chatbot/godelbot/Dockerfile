# Determined Image
FROM determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-gpu-0.19.4

WORKDIR /app
COPY ./* /app/
RUN pip install wget gradio

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_PORT=8983
ENV GRADIO_SERVER_NAME="0.0.0.0"