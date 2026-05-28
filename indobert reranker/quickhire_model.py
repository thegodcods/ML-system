# hello, disini untuk menyimpan arsitektur model reranker

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class IndoBERTRanker(nn.Module):
    """
    Ranking model architecture:

    IndoBERT
        ↓
    CLS token
        +
    Extra vectors
        ↓
       MLP
        ↓
      Score

    Args:
        model_name (str): Hugging Face IndoBERT model name.
        vector_dim (int): Dimension of additional vectors.
        hidden_dim (int): Hidden size for MLP.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        vector_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load IndoBERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        bert_hidden_size = self.encoder.config.hidden_size

        # MLP scorer
        self.mlp = nn.Sequential(
            nn.Linear(bert_hidden_size + vector_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vectors: torch.Tensor,
    ):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
            vectors: Additional feature vectors
                     Shape: (batch_size, vector_dim)

        Returns:
            score: Ranking score tensor of shape (batch_size,)
        """

        # Encode text with IndoBERT
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Concatenate CLS + external vectors
        combined = torch.cat([cls_embedding, vectors], dim=-1)

        # Predict ranking score
        score = self.mlp(combined).squeeze(-1)

        return score

# demonstrasi


"""
if __name__ == "__main__":
    MODEL_NAME = "indobenchmark/indobert-base-p1"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = IndoBERTRanker(
        model_name=MODEL_NAME,
        vector_dim=128,
        hidden_dim=256,
    )

    texts = [
        "Produk ini sangat bagus dan berkualitas.",
        "Pelayanan lambat dan mengecewakan.",
    ]

    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Example external vectors
    extra_vectors = torch.randn(len(texts), 128)

    scores = model(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        vectors=extra_vectors,
    )

    print("Scores:")
    print(scores)
"""

# jika menjalankan model sebagai dependency, import seperti:

# from quickhire_model import IndoBERTRanker
