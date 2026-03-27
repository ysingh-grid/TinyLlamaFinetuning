FROM python:3.13-slim

WORKDIR /app

# cmake + build-essential are required to compile llama-cpp-python from source.
# This layer is cached by Docker: re-runs only when this RUN step changes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (llama-cpp-python compiles here; cached until requirements.txt changes).
# Parallel C++ builds OOM HF Spaces builders (exit 137) — force single-job compile.
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV MAX_JOBS=1
ENV MAKEFLAGS=-j1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
