FROM python:3.12-slim
LABEL org.opencontainers.image.source=https://github.com/Himachiz/Unet-serving
WORKDIR /srv

# torch first, on its own layer: large, and it almost never changes.
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
CMD ["uvicorn", "app.main:api", "--host", "0.0.0.0", "--port", "8000"]
