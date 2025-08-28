# Sentiment Analysis API (PyTorch + FastAPI)


API de classificação de sentimentos com PyTorch.


## Endpoints
- `GET /health` – status e se o modelo está treinado.
- `POST /predict` – body: `{ "text": "sua frase" }` → `{ sentiment, confidence, is_trained }`.


## Executar localmente (sem Docker)
```bash
# (Windows PowerShell)
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
uvicorn app.main:app --reload --port 8000
```


## Docker
```bash
# Build
docker build -t pytorch-sentiment-api:latest .


# Run
docker run --rm -p 8000:8000 -e MODEL_WEIGHTS_PATH=models/sentiment_model.pt pytorch-sentiment-api:latest
```