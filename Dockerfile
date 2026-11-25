FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Server tanpa GPU: bangun dengan indeks PyTorch CPU agar image jauh lebih kecil, mis.
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t spotclean-ai:1.0 .
ARG TORCH_INDEX_URL=""
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
      pip install --no-cache-dir torch==2.11.0 torchvision --index-url "$TORCH_INDEX_URL"; \
    fi
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py config.py ./
COPY .streamlit .streamlit
# best.pt tidak dimasukkan ke image; dipasang sebagai volume

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
