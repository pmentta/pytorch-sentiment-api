import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
import pickle, json, os

# Garante que a raiz do projeto esteja no sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import SentimentClassifier
from train.dataset import build_dataloaders, MAX_SEQ_LEN

EPOCHS = 5
EMBED_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 2  # IMDb tem apenas 2 classes: 0=neg, 1=pos

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando em: {device}")

    # Carregar dataset e vocab
    train_loader, test_loader, vocab = build_dataloaders()
    vocab_size = len(vocab)
    pad_idx = vocab["<pad>"]
    
    print(f"Vocabulário: {vocab_size} palavras")
    
    # Inicializar modelo
    model = SentimentClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, padding_idx=pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

        # Validação
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")


    os.makedirs("models", exist_ok=True)
    with open("models/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
        hparams = {"embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM, "output_dim": OUTPUT_DIM}
    with open("models/hparams.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f)

if __name__ == "__main__":
    train_model()
