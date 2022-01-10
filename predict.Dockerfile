FROM python:3.7-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install -r requirements.txt --no-cache-dir


COPY req.txt req.txt
RUN pip install -r req.txt --no-cache-dir


COPY src/ src/
COPY data/ data/
COPY reports/ reports/
WORKDIR /
RUN mkdir /models


         

