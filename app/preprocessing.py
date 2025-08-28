# app/preprocessing.py
import re
from typing import List
import torch

# Fallback vocab/hashing params (usados se não houver vocab.pkl)
VOCAB_SIZE = 20000  # mesma const usada em seu código antigo
MAX_SEQ_LEN = 200   # recomenda-se manter igual ao usado no treino (200)

# Device exportado para InferencePipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_token_re = re.compile(r"[\w']+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    text = (text or "").lower().strip()
    return _token_re.findall(text)

def tokens_to_indices(tokens: List[str], vocab_size: int = VOCAB_SIZE) -> List[int]:
    indices = []
    for tok in tokens:
        h = (hash(tok) % (vocab_size - 1)) + 1  # 1..vocab_size-1, 0 reserved for PAD
        indices.append(h)
    return indices

def pad_or_truncate(idx: List[int], max_len: int = MAX_SEQ_LEN) -> List[int]:
    if len(idx) >= max_len:
        return idx[:max_len]
    return idx + [0] * (max_len - len(idx))

def text_to_tensor(text: str, max_len: int = MAX_SEQ_LEN, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    tokens = tokenize(text)
    indices = tokens_to_indices(tokens, vocab_size)
    padded = pad_or_truncate(indices, max_len)
    tensor = torch.tensor(padded, dtype=torch.long)
    return tensor.to(device)
