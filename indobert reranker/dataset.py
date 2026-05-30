# hello, disini menampung class dataset yang mengembalikan tensors dalam struktur rapi untuk training
# strukturisasi dataset

# batch prep yang mengembalikan input ID, attention mask, vectors, labels

# PyTorch Dataset + Batch Prep for Text + Vector Pair Comparison

from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class VectorPairDataset(Dataset):
    """
    Dataset untuk:
    - perbandingan pasangan vektor
    - label opsional
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

        text = sample["text"]

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # dynamic padding in collate_fn
            return_attention_mask=True,
            return_tensors=None,
        )

        item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "vector_a": torch.tensor(sample["vector_a"], dtype=torch.float),
            "vector_b": torch.tensor(sample["vector_b"], dtype=torch.float),
        }

        if not self.inference and "label" in sample:
            item["labels"] = torch.tensor(sample["label"], dtype=torch.long)

        return item


class VectorPairCollator:
    """
    Dynamic batch padding collator.

    Returns:
    - input_ids
    - attention_mask
    - vector_a
    - vector_b
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

        vector_a = torch.stack([x["vector_a"] for x in batch])
        vector_b = torch.stack([x["vector_b"] for x in batch])

        output = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "vector_a": vector_a,
            "vector_b": vector_b,
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

    dataset = VectorPairDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length,
        inference=inference,
    )

    collator = VectorPairCollator(tokenizer)

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
VectorPairDataset
    ↓
tokenization + tensor conversion
    ↓
VectorPairCollator
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
        "text": "this is sample one",
        "vector_a": [0.1, 0.2, 0.3],
        "vector_b": [0.5, 0.7, 0.9],
        "label": 1,
    },
    {
        "text": "another sample",
        "vector_a": [0.3, 0.4, 0.1],
        "vector_b": [0.8, 0.2, 0.6],
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
    print(batch["vector_a"].shape)
    print(batch["vector_b"].shape)
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
