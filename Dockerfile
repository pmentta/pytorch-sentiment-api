# Dockerfile for FastAPI Sentiment API
# Imagem base com Python
FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema (opcional, úteis para compilação e estabilidade)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements (sem torch)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Instala PyTorch CPU a partir do repositório oficial
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copia o restante do código
COPY . .

# Porta padrão do Uvicorn
EXPOSE 8000

# Variável opcional para caminho dos pesos do modelo
ENV MODEL_WEIGHTS_PATH="models/sentiment_model.pt"

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
