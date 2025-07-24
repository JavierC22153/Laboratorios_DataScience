FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Instalar paquetes de Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY lab2.py lab2.py

CMD [ "python", "lab2.py" ]

