# hello, disini untuk menyimpan arsitektur model reranker

import torch
import torch.nn as nn
from transformers import AutoModel


class IndoBERTRanker(nn.Module):
    """
    Ranking model architecture:

    IndoBERT
        ↓
    CLS token
        ↓
       MLP
        ↓
      Score

    Args:
        model_name (str): Hugging Face IndoBERT model name.
        hidden_dim (int): Hidden size for MLP.
        dropout (float): Dropout rate.
    """

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_bert=True
    ):
        super().__init__()

        # Load IndoBERT encoder
        self.encoder = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
            )

        bert_hidden_size = self.encoder.config.hidden_size

        # MLP scorer
        self.mlp = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        if freeze_bert:
            self.freeze_encoder()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)

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

        # Predict ranking score
        score = self.mlp(cls_embedding).squeeze(-1)

        return score

# demonstrasi


"""
if __name__ == "__main__":
    MODEL_NAME = "indobenchmark/indobert-base-p1"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = IndoBERTRanker(
        model_name=MODEL_NAME,
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

    scores = model(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
    )

    print("Scores:")
    print(scores)
"""

# jika menjalankan model sebagai dependency, import seperti:

# from quickhire_model import IndoBERTRanker
