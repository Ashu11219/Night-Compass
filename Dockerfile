FROM python:3.12-slim-bookworm

# ultralytics pulls in a version of OpenCV that needs these two system libraries.
# The slim image does not have them, so the app would crash on "import cv2" without this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only build of PyTorch first (a few hundred MB instead of several GB).
# Cloud Run has no GPU, so the GPU build would only waste space and startup time.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
