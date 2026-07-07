FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Writable dirs for the trace DB and the ONNX embedding-model cache
# (HF Spaces runs the container as a non-root user).
RUN mkdir -p /app/data /cache && chmod -R 777 /app/data /cache
ENV XDG_CACHE_HOME=/cache

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
