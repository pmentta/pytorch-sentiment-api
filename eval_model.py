import torch, pickle
from sklearn.metrics import classification_report, confusion_matrix
from train.dataset import build_dataloaders, MAX_SEQ_LEN
from app.model import SentimentClassifier
import numpy as np

# carregar vocab salvo e dataloaders com esse vocab
vocab = pickle.load(open("models/vocab.pkl","rb"))
train_dl, test_dl, _ = build_dataloaders(vocab=vocab)

# inferir hyperparams do checkpoint (ou ajuste manual)
state = torch.load("models/sentiment_model.pt", map_location="cpu")
emb = state[[k for k in state.keys() if k.endswith("embedding.weight")][0]]
vocab_size, embed_dim = emb.shape
# inferir hidden_dim via lstm.weight_hh_l0 shape
lstm_whh = state[[k for k in state.keys() if k.endswith("lstm.weight_hh_l0")][0]]
hidden_dim = lstm_whh.shape[1]
fc_w = state[[k for k in state.keys() if k.endswith("fc.weight")][0]]
output_dim = fc_w.shape[0]

print("Inferred:", vocab_size, embed_dim, hidden_dim, output_dim)

pad_idx = vocab.get("<pad>", 0)
model = SentimentClassifier(vocab_size, embed_dim, hidden_dim, output_dim, padding_idx=pad_idx)
model.load_state_dict(state)
model.eval()

y_true, y_pred, y_probs = [], [], []
misclassified = []

with torch.no_grad():
    for texts, labels in test_dl:
        logits = model(texts)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.numpy().tolist())
        y_probs.extend(probs[:,1].numpy().tolist())  # prob positive

# report
print(classification_report(y_true, y_pred, target_names=["negative","positive"]))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

# show some false negatives/positives for inspection
print("\nMost confident wrong predictions (example indexes omitted):")
# we'll scan test set again to print text for some misclassified samples
count = 0
for batch in test_dl:
    texts, labels = batch
    with torch.no_grad():
        logits = model(texts)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    for i in range(len(labels)):
        if preds[i].item() != labels[i].item():
            print("---")
            print("Label:", labels[i].item(), "Pred:", preds[i].item(), "Prob:", probs[i].tolist())
            # text retrieval requires original dataset; we can reconstruct from test_dl.dataset
            try:
                raw_text = test_dl.dataset[i]["text"]
                print("Text:", raw_text[:300])
            except Exception:
                print("Text unavailable in this loader view")
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break