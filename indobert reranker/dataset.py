# hello, disini menampung class dataset yang mengembalikan tensors dalam struktur rapi untuk training
# strukturisasi dataset

# batch prep yang mengembalikan input ID, attention mask, vectors, labels

# PyTorch Dataset + Batch Prep for Text + Vector Pair Comparison

from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class RerankerDataset(Dataset):
    """
    Dataset untuk:
    - perbandingan reranker

    Input:
    - query
    - dokumen
    - vector dokumen

    output:
    - input_ids
    - attention_mask
    - document_vector
    - labels opsional
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_length: int = 128,
        inference: bool = False,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inference = inference

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        query = sample["query"]
        document = sample["document"]

        encoded = self.tokenizer(
            query,
            document,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # dynamic padding in collate_fn
            return_attention_mask=True,
            return_tensors=None,
        )

        item = {
            "input_ids": torch.tensor(encoded["input_ids"],
                                      dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"],
                                           dtype=torch.long)
        }

        if not self.inference and "label" in sample:
            item["labels"] = torch.tensor(sample["label"],
                                          dtype=torch.float)

        return item


class RerankerCollator:
    """
    Dynamic batch padding collator.

    Returns:
    - input_ids
    - attention_mask
    - document vector
    - labels (optional)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [x["input_ids"] for x in batch]
        attention_masks = [x["attention_mask"] for x in batch]

        padded = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
            },
            padding=True,
            return_tensors="pt",
        )

        document_vector = torch.stack([x["document_vector"] for x in batch])

        output = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "document_vector": document_vector
        }

        if "labels" in batch[0]:
            output["labels"] = torch.stack([x["labels"] for x in batch])

        return output

# helper untuk konsistensi data terlepas dari training/infer


def build_dataloader(
    samples,
    model_name: str = "indobenchmark/indobert-base-p1",
    batch_size: int = 8,
    shuffle: bool = False,
    max_length: int = 256,
    inference: bool = False,
    num_workers: int = 0,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = RerankerDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length,
        inference=inference,
    )

    collator = RerankerCollator(tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader

# Example Usage

# Training


"""
pipeline:

samples list
    ↓
RerankerDataset
    ↓
tokenization + tensor conversion
    ↓
RerankerCollator
    ↓
dynamic batch padding
    ↓
DataLoader
    ↓
training loop
"""

"""
train_samples = [
    {
        "query": "cara membuat sambal",
        "document": "Resep sambal terasi...",
        "document_vector": [0.1] * 384,
        "label": 1,
    },
    {
        "query": "cara menyetrika uang",
        "document": "setrika merek maspion",
        "document_vector": [0.8] * 384,
        "label": 0,
    },
]

train_loader = build_dataloader(
    train_samples,
    batch_size=2,
    shuffle=True,
)

for batch in train_loader:
    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)
    print(batch["document_vector"].shape)
    print(batch["labels"].shape)
"""

# Inference

"""
infer_samples = [
    {
        "text": "new unseen example",
        "vector_a": [0.4, 0.1, 0.7],
        "vector_b": [0.2, 0.9, 0.5],
    }
]

infer_loader = build_dataloader(
    infer_samples,
    batch_size=1,
    inference=True,
)

for batch in infer_loader:
    print(batch.keys())
"""
