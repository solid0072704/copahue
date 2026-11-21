# Tu Cartera: imagen para publicar el servidor (Render, Railway, Fly.io o cualquier servicio con Docker).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TUCARTERA_DB=/data/tucartera.db
WORKDIR /app
COPY requirements-tucartera.txt ./
RUN pip install --no-cache-dir -r requirements-tucartera.txt
COPY tucartera ./tucartera
RUN mkdir -p /data && useradd --create-home --uid 10001 tucartera && chown tucartera /data
USER tucartera
EXPOSE 8000
CMD ["python", "-m", "tucartera", "servir", "--host", "0.0.0.0", "--puerto", "8000"]
