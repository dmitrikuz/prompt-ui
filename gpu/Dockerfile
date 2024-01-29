FROM python:3.10

COPY gpu/requirements.txt /app/
 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r /app/requirements.txt

COPY . /app/

WORKDIR /app/
