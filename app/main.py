from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .inference import InferencePipeline

app = FastAPI(title="Sentiment Analysis API (PyTorch)", version="0.1.0")
pipe = InferencePipeline()

class InputText(BaseModel):
    text: str


@app.get("/health")
async def health():
    return {"status": "ok", "is_trained": pipe.is_trained}


@app.post("/predict")
async def predict(inp: InputText):
    if not pipe.is_trained:
        # Permite rodar, mas é melhor sinalizar claramente que o modelo não foi treinado ainda
        # Você pode mudar para HTTP 503 se preferir bloquear a inferência antes de treinar.
        result = pipe.predict(inp.text)
        result["warning"] = (
        "Modelo ainda não treinado. Treine em 'train/train.py' e gere 'models/sentiment_model.pt'."
        )
        return result


    try:
        return pipe.predict(inp.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))