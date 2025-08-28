# scripts/test_inference_local.py
import torch, pickle
from app.model import SentimentClassifier
from train.dataset import tokenizer, MAX_SEQ_LEN

# carregar vocab
vocab = pickle.load(open("models/vocab.pkl","rb"))
vocab_size = len(vocab)
pad_idx = vocab.get("<pad>", 0)

# inferir hparams (ou ajuste manual se souber)
EMBED_DIM = 128   # ajuste se souber do checkpoint
HIDDEN_DIM = 256  # ajuste se souber do checkpoint
OUTPUT_DIM = 2

model = SentimentClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, padding_idx=pad_idx)
model.load_state_dict(torch.load("models/sentiment_model.pt", map_location="cpu"))
model.eval()

def process(text):
    tokens = tokenizer(text)
    idx = [vocab.get(t, vocab.get("<unk>", 1)) for t in tokens][:MAX_SEQ_LEN]
    if len(idx) < MAX_SEQ_LEN:
        idx += [vocab.get("<pad>", 0)] * (MAX_SEQ_LEN - len(idx))
    return torch.tensor(idx, dtype=torch.long).unsqueeze(0)

for s in ["I loved that movie", "I hated that movie", "000000000000000000", "it was ok, nothing special"]:
    x = process(s)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    print(s, "->", probs.detach().numpy())
