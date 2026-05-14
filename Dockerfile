# Imagen Base
FROM public.ecr.aws/docker/library/python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENTRYPOINT ["python", "-m", "src.api_inference"]
CMD ["--config", "inference.yaml"]
