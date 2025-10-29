FROM python:3.10-slim

WORKDIR /app

# Instala dependencias del sistema para pandas y GCP
RUN apt-get update && apt-get install -y gcc g++ libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app_online_pozos.py"]

