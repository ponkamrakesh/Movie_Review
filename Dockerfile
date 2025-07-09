# ✅ Base Image
FROM python:3.10-slim

# ✅ Set working directory
WORKDIR /app

# ✅ Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ✅ Copy project files
COPY . /app

# ✅ Install Python dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ✅ Expose the port (FastAPI runs on 7860 by HF default)
EXPOSE 7860

# ✅ Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
