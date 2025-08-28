# scripts/save_vocab_from_dataset.py
from train.dataset import build_dataloaders
import pickle, os

# irá construir vocab a partir do dataset (sem treinar)
_, _, vocab = build_dataloaders()
os.makedirs("models", exist_ok=True)
with open("models/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
print("vocab salvo em models/vocab.pkl; len =", len(vocab))
