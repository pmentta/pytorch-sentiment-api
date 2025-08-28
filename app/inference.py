# app/inference.py
import os
import json
import pickle
from typing import Dict, List

import torch
import torch.nn.functional as F

from .model import SentimentClassifier

# caminhos padrão
WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "models/sentiment_model.pt")
VOCAB_PATH = os.getenv("MODEL_VOCAB_PATH", "models/vocab.pkl")
HPARAMS_PATH = os.getenv("MODEL_HPARAMS_PATH", "models/hparams.json")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels default (serão ajustadas se checkpoint tiver outra dimensão)
DEFAULT_LABELS = ["negative", "positive"]


def _find_key_ending(state: Dict[str, torch.Tensor], suffix: str):
    for k in state.keys():
        if k.endswith(suffix):
            return k
    return None


def _safe_torch_load(path: str, map_location):
    """
    Tenta carregar com weights_only=True quando disponível (mais seguro).
    Fallback para carregar sem weights_only.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # versão antiga do PyTorch que não tem weights_only
        return torch.load(path, map_location=map_location)


class InferencePipeline:
    def __init__(self):
        self.device = device
        self.vocab = None
        self.use_vocab = False
        self.is_trained = False
        self.model = None
        self.labels = DEFAULT_LABELS

        # se existir vocab salvo, carregue
        if os.path.exists(VOCAB_PATH):
            try:
                with open(VOCAB_PATH, "rb") as f:
                    self.vocab = pickle.load(f)
                self.use_vocab = True
            except Exception:
                self.vocab = None
                self.use_vocab = False

        # se hparams salvo, carregue (opcional)
        saved_hparams = None
        if os.path.exists(HPARAMS_PATH):
            try:
                with open(HPARAMS_PATH, "r", encoding="utf-8") as f:
                    saved_hparams = json.load(f)
            except Exception:
                saved_hparams = None

        # se existe checkpoint, deduz arquitetura e carregue
        if os.path.exists(WEIGHTS_PATH):
            state = _safe_torch_load(WEIGHTS_PATH, map_location=self.device)
            # se o arquivo salvo for um state_dict (dict de tensores)
            if isinstance(state, dict):
                # tenta encontrar as chaves comuns e deduzir shapes
                emb_key = _find_key_ending(state, "embedding.weight")
                lstm_wih_key = _find_key_ending(state, "lstm.weight_ih_l0")
                lstm_whh_key = _find_key_ending(state, "lstm.weight_hh_l0")
                fc_w_key = _find_key_ending(state, "fc.weight")

                if emb_key and lstm_wih_key and lstm_whh_key and fc_w_key:
                    emb_shape = tuple(state[emb_key].shape)
                    lstm_wih_shape = tuple(state[lstm_wih_key].shape)
                    lstm_whh_shape = tuple(state[lstm_whh_key].shape)
                    fc_w_shape = tuple(state[fc_w_key].shape)

                    inferred_vocab_size = emb_shape[0]
                    inferred_embed_dim = emb_shape[1]
                    # weight_ih shape is (4*hidden_dim, input_size) so input_size = embed_dim
                    inferred_hidden_dim = lstm_whh_shape[1]
                    inferred_output_dim = fc_w_shape[0]

                    pad_idx = 0
                    if self.use_vocab and isinstance(self.vocab, dict) and "<pad>" in self.vocab:
                        pad_idx = self.vocab["<pad>"]

                    # instancia modelo com os tamanhos inferidos
                    self.model = SentimentClassifier(
                        vocab_size=inferred_vocab_size,
                        embed_dim=inferred_embed_dim,
                        hidden_dim=inferred_hidden_dim,
                        output_dim=inferred_output_dim,
                        padding_idx=pad_idx
                    ).to(self.device)

                    # carrega state dict
                    try:
                        self.model.load_state_dict(state)
                        self.model.eval()
                        self.is_trained = True
                        # ajusta labels dinamicamente
                        if inferred_output_dim == 2:
                            self.labels = ["negative", "positive"]
                        elif inferred_output_dim == 3:
                            self.labels = ["negative", "neutral", "positive"]
                        else:
                            # generic labels
                            self.labels = [f"class_{i}" for i in range(inferred_output_dim)]
                    except Exception as e:
                        # carregamento falhou — cadastrar o erro em detalhe durante inicialização
                        print("Erro ao carregar state_dict:", e)
                        self.model = None
                        self.is_trained = False
                else:
                    print("State dict não contém chaves esperadas (embedding/lstm/fc).")
            else:
                print("Checkpoint não é um state_dict esperado. Re-save como model.state_dict().")

        # se não houve checkpoint compatível, cria modelo leve fallback (não treinado)
        if self.model is None:
            # fallback: se existe vocab, usa seu tamanho, senão usa 20k
            fallback_vocab_size = len(self.vocab) if self.use_vocab and isinstance(self.vocab, dict) else 20000
            self.model = SentimentClassifier(vocab_size=fallback_vocab_size).to(self.device)
            self.model.eval()
            self.is_trained = False

    @torch.inference_mode()
    def predict(self, text: str) -> Dict[str, object]:
        # tokenização e transformação em tensor:
        if self.use_vocab and isinstance(self.vocab, dict):
            # usa same tokenizer/logic do treino: token -> idx via vocab
            from train.dataset import tokenizer, MAX_SEQ_LEN  # usa tokenizer que você usou no treino
            tokens = tokenizer(text)
            indices = [ self.vocab.get(t, self.vocab.get("<unk>", 1)) for t in tokens ][:MAX_SEQ_LEN]
            if len(indices) < MAX_SEQ_LEN:
                indices += [ self.vocab.get("<pad>", 0) ] * (MAX_SEQ_LEN - len(indices))
            x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        else:
            # fallback hashing trick (se você não salvou vocab)
            from .preprocessing import text_to_tensor
            x = text_to_tensor(text).unsqueeze(0).to(self.device)

        logits = self.model(x)  # (1, C) logits
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

        return {
            "sentiment": self.labels[pred_idx.item()],
            "confidence": float(confidence.item()),
            "is_trained": self.is_trained,
        }
