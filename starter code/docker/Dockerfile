FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install --upgrade pip
ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /code