FROM python:3.11.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install setuptools versi lama DULU, lalu install whisper di sini
RUN pip install "setuptools==67.8.0"
RUN pip install openai-whisper==20231117

# Sekarang baru install sisanya (boleh upgrade setuptools)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
