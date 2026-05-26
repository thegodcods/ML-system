# hello, ini untuk formatting jobdesc
# dan mempertahankan pseudo structure pada CV
# salah satu fungsi penting adalah pair assembly

# dibawah ini harusnya ada sequence formatting policy

# dibawah ini harusnya ada tensor validation

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PairRecord:
    """
    Satu PairRecord berisi pasangan sample.
    Satu dari hasil parsing CV dan Vektor CV
    Satu lagi dari parsing Jobdesc dan Vektor Jobdesc

    Example:
            PairRecord(
            cv_parsed="hasil parsing CV",
            jobdesc_parsed="hasil parsing job desc",
            cv_vector=np.array([0.1, 0.2]),
            jobdesc_vector=np.array([0.3, 0.4]),
            label=1,
        )
    """
    cv_text: Optional[str] = None
    jobdesc_text: Optional[str] = None

    cv_vector: Optional[np.ndarray] = None
    jobdesc_vector: Optional[np.ndarray] = None

    label: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PairAssembler:
    """
    Flexible pair assembler untuk:
        - raw text
        - vector embeddings
        - labels / metadata

    Main goals:
        1. memudahkan pembuatan pair data
        2. memvalidasi dimensi pair.
        3. Export batches untuk ML pipelines.
        4. untuk kegiatan similarity, ranking dan tugas perbandingan kontras
    """
    # mendefinisikan self sebagai internal storage
    def __init__(self, vector_dtype=np.float32):
        self.vector_dtype = vector_dtype
        self.records: List[PairRecord] = []

    # layer tambah data

    # function ini menambah pair ke assembler
    # note untuk dev: truncation/pemotongan data bisa dilakukan untuk teks
    # agar model embedding tidak memotong token penting
    def add_pair(
        self,
        cv_text: Optional[str] = None,
        jobdesc_text: Optional[str] = None,
        cv_vector: Optional[Sequence[float]] = None,
        jobdesc_vector: Optional[Sequence[float]] = None,
        label: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PairRecord:
        """
        menambah pair ke assembler
        """
        cv_vec = self._to_vector(cv_vector)
        jobdesc_vec = self._to_vector(jobdesc_vector)

        # baris 1: pemeriksaan kekosongan vector CV dan vector jobdesc
        # baris 2: pemeriksaan kesamaan shape vector
        # baris 2 tidak akan error selama model embedding CV
        # dan embedding jobdesc sama
        if cv_vec is not None and jobdesc_vec is not None:
            if cv_vec.shape != jobdesc_vec.shape:
                raise ValueError(f"dimensi vektor tidak sama:"
                                 f"{cv_vec.shape} != {jobdesc_vec.shape}"
                                 )

    # menggunakan PairRecord untuk mapping function dalam memory ke
        record = PairRecord(
                cv_text=cv_text,
                jobdesc_text=jobdesc_text,
                cv_vector=cv_vec,
                jobdesc_vector=jobdesc_vec,
                label=label,
                metadata=metadata or {},
            )

        self.records.append(record)
        return record

    # batch export

    def to_dict(self) -> Dict[str, List[Any]]:
        """
        Mengubah semua record yang masuk menjadi dictionary.
        berguna untuk training pipeline.
        """

        return {
            "cv_texts": [r.cv_text for r in self.records],
            "jobdesc_texts": [r.jobdesc_text for r in self.records],
            "cv_vectors": [r.cv_vector for r in self.records],
            "jobdesc_vectors": [r.jobdesc_vector for r in self.records],
            "labels": [r.label for r in self.records],
            "metadata": [r.metadata for r in self.records],
        }

    def vector_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mengexport vector menjadi dua matrix yang tersusun
        """
        cv = []
        jobdesc = []

        for r in self.records:
            if r.cv_vector is None or r.jobdesc_vector is None:
                continue
            cv.append(r.cv_vector)
            jobdesc.append(r.jobdesc_vector)
        if not cv:
            raise ValueError("Tidak ada pasangan vektor yang tersedia")

        return np.vstack(cv), np.vstack(jobdesc)

    # fungsi cosine similarity dengan torch

    def cosine_similarities(self):

        scores = []

        for r in self.records:

            if r.cv_vector is None or r.jobdesc_vector is None:
                continue

            left = torch.tensor(r.cv_vector)
            right = torch.tensor(r.jobdesc_vector)

            sim = F.cosine_similarity(
                left.unsqueeze(0),
                right.unsqueeze(0)
            )

            scores.append(sim.item())

        return scores
    # internal helper

    def _to_vector(
        self,
        vector: Optional[Sequence[float]],
    ) -> Optional[np.ndarray]:
        if vector is None:
            return None

        return np.asarray(vector, dtype=self.vector_dtype)

# cth penggunaan


"""
from preprocess import PairAssembler

if __name__ == "__main__":
    assembler = PairAssembler()

    assembler.add_pair(
        cv_text="machine learning",
        jobdesc_text="artificial intelligence",
        cv_vector=[0.1, 0.4, 0.2],
        jobdesc_vector=[0.15, 0.35, 0.22],
        label=1,
        metadata={"source": "training"},
    )

    assembler.add_pair(
        cv_text="cat",
        jobdesc_text="car engine",
        cv_vector=[0.9, 0.1, 0.0],
        jobdesc_vector=[0.1, 0.7, 0.9],
        label=0,
    )

    print("\n=== DICTIONARY EXPORT ===")
    print(assembler.to_dict())

    print("\n=== VECTOR MATRICES ===")
    cv_matrix, jobdesc_matrix = assembler.vector_matrix()
    print(cv_matrix)
    print(jobdesc_matrix)

    print("\n=== COSINE SIMILARITIES ===")
    print(assembler.cosine_similarities())

# bisa juga menggunakan for loop
file = [
    {
        "cv_text": "Python engineer",
        "jobdesc_text": "Backend developer",
        "cv_vector": [0.1, 0.2, 0.3],
        "jobdesc_vector": [0.1, 0.21, 0.29],
        "label": 1
    },

    {
        "cv_text": "Accountant",
        "jobdesc_text": "Machine learning scientist",
        "cv_vector": [0.9, 0.1, 0.2],
        "jobdesc_vector": [0.1, 0.8, 0.9],
        "label": 0
    }
]

for entry in file:
    cv_text = entry["cv_text"]
    jobdesc_text = entry["jobdesc_text"]
    cv_vector = entry["cv_vector"]
    jobdesc_vector = entry["jobdesc_vector"]
    label = entry["label"]

    assembler.add_pair(
        cv_text=cv_text,
        jobdesc_text=jobdesc_text,
        cv_vector=cv_vector,
        jobdesc_vector=jobdesc_vector
    )
"""
