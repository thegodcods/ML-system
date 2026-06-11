# hello, ini untuk formatting jobdesc
# dan mempertahankan pseudo structure pada CV
# salah satu fungsi penting adalah pair assembly

# dibawah ini harusnya ada sequence formatting policy

# dibawah ini harusnya ada tensor validation

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RerankerRecord:
    """
    query berasal dari query template/query custom dari user


    dibuat dengan tujuan menstandarisasi input yang masuk

    Example:
            PairRecord(
            query="butuh developer front end",
            document="hasil parsing CV",
            document_vector=np.array([0.1, 0.2]),,
            label=1,
        )
    """
    query: str
    document: str
    document_vector: Optional[np.ndarray] = None
    label: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RerankerAssembler:
    """
    Flexible reranker assembler untuk:
        - raw text
        - vector embeddings
        - labels / metadata

    Main goals:
        1. validasi skema
        2. validasi vektor
        3. sample assembly
    """
    # mendefinisikan self sebagai internal storage
    def __init__(self,
                 vector_dim: Optional[int] = None,
                 vector_dtype=np.float32):
        self.vector_dim = vector_dim
        self.vector_dtype = vector_dtype
        self.records: List[RerankerRecord] = []

    # layer tambah data

    # function ini menambah pair ke assembler
    # note untuk dev: truncation/pemotongan data bisa dilakukan untuk teks
    # agar model embedding tidak memotong token penting
    def add_sample(
        self,
        query: Optional[str] = None,
        document: Optional[str] = None,
        document_vector: Optional[Sequence[float]] = None,
        label: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RerankerRecord:
        """
        menambah pair ke assembler
        """
        if not query:
            raise ValueError("query kosong")

        if not document:
            raise ValueError("document kosong")

        vector = self._to_vector(document_vector)

        # baris 1: pemeriksaan kekosongan vector CV dan vector jobdesc
        # baris 2: pemeriksaan kesamaan shape vector
        # baris 2 tidak akan error selama model embedding CV
        # dan embedding jobdesc sama
        if vector is not None and self.vector_dim is not None:
            if vector.shape[0] != self.vector_dim:
                raise ValueError(
                    f"vector dimension mismatch: "
                    f"{vector.shape[0]} != {self.vector_dim}"
                    )

    # menggunakan PairRecord untuk mapping function dalam memory ke
        record = RerankerRecord(
                query=query,
                document=document,
                document_vector=document_vector,
                label=label,
                metadata=metadata or {},
            )

        self.records.append(record)
        return record

    def to_samples(self) -> List[Dict[str, Any]]:
        """
        Export format compatible with dataset.py
        """

        samples = []

        for r in self.records:

            sample = {
                "query": r.query,
                "document": r.document,
                "document_vector": r.document_vector,
            }

            if r.label is not None:
                sample["label"] = r.label

            if r.metadata:
                sample["metadata"] = r.metadata

            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.records)

    # internal helper

    def _to_vector(
        self,
        vector: Optional[Sequence[float]],
    ) -> Optional[np.ndarray]:
        if vector is None:
            return None

        return np.asarray(vector, dtype=self.vector_dtype)

# cth penggunaan

'''
from dataset import build_dataloader

assembler = RerankerAssembler(vector_dim=384)

assembler.add_sample(
    query="Looking for a Python backend engineer",
    document="""
    John Doe

    Skills:
    Python
    FastAPI
    PostgreSQL

    Experience:
    Backend Engineer 4 years
    """,
    document_vector=[0.1] * 384,
)

assembler.add_sample(
    query="Looking for a Python backend engineer",
    document="""
    Jane Smith

    Skills:
    Accounting
    Tax
    Finance

    Experience:
    Senior Accountant
    """,
    document_vector=[0.2] * 384,
)

samples = assembler.to_samples()

print(samples[0])
print(len(assembler))
print(samples[0].keys())

for s in samples:
    print(s["document_vector"] is None)

loader = build_dataloader(
    samples=samples,
    batch_size=2,
    shuffle=False,
)

for batch in loader:

    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)
    print(batch["document_vector"].shape)

    break
'''
