FROM python:3.11.12-slim

WORKDIR /app

# Install setuptools dulu di sistem, SEBELUM venv apapun
RUN pip install "setuptools>=68.0.0"

# Copy requirements dan install dengan no-build-isolation
COPY requirements.txt .
RUN pip install --no-build-isolation -r requirements.txt

# Copy semua kode
COPY . .

# Jalankan server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
