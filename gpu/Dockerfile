FROM python:3.10

COPY gpu/requirements.txt /app/
 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app/

WORKDIR /app/
