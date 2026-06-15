FROM python:3.11.12-slim

WORKDIR /app

# Install git (untuk install openai-whisper dari GitHub) dan setuptools
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install "setuptools>=68.0.0"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
