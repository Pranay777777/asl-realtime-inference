# The live demo (space/app.py) on CPU. Models are not in git; mount them at run time:
#   docker build -t asl-demo .
#   docker run --rm -p 7860:7860 -v "$PWD/models:/app/models:ro" asl-demo   ->  http://localhost:7860
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860
WORKDIR /app
COPY space/requirements.txt .
RUN pip install -r requirements.txt gradio==6.29.0
COPY space/app.py .

RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 7860
CMD ["python", "app.py"]
