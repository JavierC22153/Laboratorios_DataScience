FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app

COPY requirements.txt .

# Instalar paquetes de Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY lab1.py lab1.py

CMD [ "python", "lab1.py" ]

