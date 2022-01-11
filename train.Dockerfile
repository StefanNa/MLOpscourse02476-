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
RUN pip install dvc[gs]

RUN mkdir /app
COPY src/ /app/src/
COPY .dvc /app/.dvc
# COPY .git /app/.git
COPY data.dvc /app/data.dvc
RUN mkdir /app/data && mkdir /app/reports
WORKDIR /app
RUN dvc config core.no_scm true
RUN dvc pull

ENTRYPOINT ["python", "-u", "src/models/train_model.py", "train", "--lr=0.003"]

