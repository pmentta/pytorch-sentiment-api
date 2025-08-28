# train/dataset.py
import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter
from typing import Dict, Tuple

BATCH_SIZE = 64
MAX_SEQ_LEN = 200
MAX_VOCAB_SIZE = 30000  # opcional: limite para o vocabulário

tokenizer = get_tokenizer("basic_english")


def build_vocab_from_dataset(dataset, max_vocab_size: int = MAX_VOCAB_SIZE) -> Dict[str, int]:
    """
    Constrói um token -> idx dict simples.
    Índices reservados:
      <pad> -> 0
      <unk> -> 1
    Tokens frequentes começam em 2.
    """
    counter = Counter()
    for example in dataset:
        tokens = tokenizer(example["text"])
        counter.update(tokens)

    most_common = counter.most_common(max_vocab_size)
    token2idx = {"<pad>": 0, "<unk>": 1}
    for token, _ in most_common:
        if token not in token2idx:
            token2idx[token] = len(token2idx)
    return token2idx


def process_text_to_tensor(text: str, vocab: Dict[str, int], max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    tokens = tokenizer(text)
    indices = [vocab.get(t, vocab["<unk>"]) for t in tokens][:max_len]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)


def collate_batch_factory(vocab: Dict[str, int]):
    def collate_batch(batch):
        texts = []
        labels = []
        for example in batch:
            texts.append(process_text_to_tensor(example["text"], vocab))
            labels.append(example["label"])  # IMDb: 0 = neg, 1 = pos
        return torch.stack(texts), torch.tensor(labels, dtype=torch.long)
    return collate_batch


def build_dataloaders(vocab: Dict[str, int] = None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, int]]:
    """
    Retorna: train_dataloader, test_dataloader, vocab (dict token->idx)
    """
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if vocab is None:
        vocab = build_vocab_from_dataset(train_dataset)

    collate_fn = collate_batch_factory(vocab)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader, vocab
