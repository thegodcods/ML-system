# hello, ini untuk formatting jobdesc
# dan mempertahankan pseudo structure pada CV
# salah satu fungsi penting adalah record assembly

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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


class TextStructurer:
    """
    Converts cleaned but unstructured CV / jobdesc into
    model-ready structured text.
    """

    def _create_pseudo_lines(self, text):

        separators = [
            "work experience",
            "professional experience",
            "technical skills",
            "skills",
            "education",
            "projects",
            "about me",
            "summary",
            "contact"
        ]

        lowered = text.lower()

        for sep in separators:
            lowered = lowered.replace(
                sep,
                f"\n{sep}\n"
            )

        return [
            t.strip()
            for t in lowered.split("\n")
            if t.strip()
        ]

    def structure_job(self, text: str):

        lines = self._create_pseudo_lines(text)

        return (
            "[JOB]\n"
            + "\n".join(lines[:80])
        )

    def structure_resume(self, text: str) -> str:
        """
        Convert messy CV into structured schema
        """

        sections = self._split_sections(text)

        ordered = [
            ("TITLE", sections.get("title", [])),
            ("SUMMARY", sections.get("summary", [])),
            ("SKILLS", sections.get("skills", [])),
            ("EXPERIENCE", sections.get("experience", [])),
            ("EDUCATION", sections.get("education", [])),
        ]

        formatted = "[RESUME]\n"
        for k, v in ordered:
            if v:
                formatted += f"{k}:\n{self._join(k, v)}\n\n"
        return formatted.strip()

    def _extract_title(self, lines):
        title_keywords = [
            "engineer", "developer", "manager",
            "analyst", "architect", "designer"
        ]

        title_parts = []

        for line in lines[:6]:  # top region only
            lowered = line.lower()
            word_count = len(line.split())

            if (any(k in lowered for k in title_keywords) and word_count < 4):
                title_parts.append(line)

        return "".join(title_parts)

    def _split_sections(self, text: str):
        """
        state machine style heuristic section detection
        backend already cleaned text → no regex cleanup needed
        """

        lines = self._create_pseudo_lines(text)
        lines_lower = [l.lower() for l in lines]

        sections = {
            "title": [],
            "summary": [],
            "skills": [],
            "experience": [],
            "education": [],
            "other": []
        }

        title = self._extract_title(lines)

        if title:
            sections["title"] = [title]

        current = "other"

        for original, line in zip(lines, lines_lower):

            detected = self._detect_section(line)

            if detected:
                current = detected
                continue

            sections[current].append(original)

        """print("=== SEGMENTS ===")
        for k, v in sections.items():
            print(k, ":", v[:2])"""

        return sections

    def _detect_section(self, line: str):

        section_map = {
            "summary": [
                "about me",
                "summary",
                "profile",
                "objective"
            ],
            "skills": [
                "skills",
                "technical skills",
                "competencies"
            ],
            "experience": [
                "experience",
                "work experience",
                "employment"
            ],
            "education": [
                "education",
                "academic"
            ]
        }

        for section, keys in section_map.items():
            for k in keys:
                if k in line:
                    return section

        return None

    def _truncate_words(
            self,
            text: str,
            max_words: int
    ):

        words = text.split()

        return " ".join(words[:max_words])

    def _join(self, section_name, items):
        SECTION_LIMITS = {
            "TITLE": 20,
            "SUMMARY": 80,
            "SKILLS": 80,
            "EXPERIENCE": 180,
            "EDUCATION": 50
        }

        limit = SECTION_LIMITS.get(
            section_name,
            50
        )

        return self._truncate_words(" ".join(items), limit)

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
