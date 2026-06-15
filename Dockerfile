FROM python:3.11.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Downgrade setuptools ke versi yang masih include pkg_resources
RUN pip install "setuptools==67.8.0"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
