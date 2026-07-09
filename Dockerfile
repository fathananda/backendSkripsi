FROM python:3.11.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install pkg_resources via package terpisah yang tersedia di PyPI
RUN pip install pkginfo setuptools==67.8.0 wheel

# Install whisper dengan no-build-isolation agar pakai setuptools yang sudah ada
RUN pip install --no-build-isolation openai-whisper==20231117 numpy==1.26.4

# Install sisanya
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
