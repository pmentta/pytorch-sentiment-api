# app/model.py
import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 128, output_dim: int = 2, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # NÃO use Softmax aqui — CrossEntropyLoss espera logits brutos.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) ints (token ids)
        Retorna logits (batch, output_dim)
        """
        embedded = self.embedding(x)                    # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)            # hidden: (num_layers * num_directions, batch, hidden_dim)
        logits = self.fc(hidden[-1])                    # pega o último layer (batch, hidden_dim) -> (batch, output_dim)
        return logits
