FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Container paths. Keep runtime code, model/templates, videos, and outputs predictable.
ENV APP_HOME=/workspace/Sugery
ENV PYTHONPATH=/workspace/Sugery:/workspace/Sugery/src
ENV SURGERY_BASE_DIR=/workspace/Sugery
ENV SURGERY_TEST_VIDEO_BASE=/data/videos
ENV SURGERY_OUTPUT_BASE_DIR=/workspace/outputs

RUN apt-get update && apt-get install -y --no-install-recommends     libgl1     libglib2.0-0     libsm6     libxext6     libxrender-dev     ffmpeg     git     && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Sugery
RUN mkdir -p /data/videos /workspace/outputs && chmod 777 /data/videos /workspace/outputs

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8000

CMD ["python", "src/main_parallel.py"]
