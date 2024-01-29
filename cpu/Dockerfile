FROM python:3.10

COPY cpu/requirements.txt /app/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip 
RUN pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install -r /app/requirements.txt

COPY . /app/

WORKDIR /app/

