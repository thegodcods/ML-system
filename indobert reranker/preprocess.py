# hello, ini untuk formatting jobdesc
# dan mempertahankan pseudo structure pada CV
# salah satu fungsi penting adalah record assembly

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json

@dataclass
class RerankerRecord:
    """
    query berasal dari query template/query custom dari user


    dibuat dengan tujuan menstandarisasi input yang masuk

    Example:
            PairRecord(
            query="butuh developer front end",
            document="hasil parsing CV",
            label=1,
        )
    """
    query: str
    document: str
    label: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RerankerAssembler:
    """
    Flexible reranker assembler untuk:
        - raw text query dan dokumen
        - labels / metadata

    Main goals:
        1. validasi skema
        2. validasi vektor
        3. sample assembly
    """
    # mendefinisikan self sebagai internal storage
    def __init__(self):
        self.records: List[RerankerRecord] = []

    # layer tambah data

    # function ini menambah pair ke assembler
    # note untuk dev: truncation/pemotongan data bisa dilakukan untuk teks
    # agar model embedding tidak memotong token penting
    def add_sample(
        self,
        query: Optional[str] = None,
        document: Optional[str] = None,
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

    # menggunakan PairRecord untuk mapping function dalam memory ke
        record = RerankerRecord(
                query=query,
                document=document,
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
            }

            if r.label is not None:
                sample["label"] = r.label

            if r.metadata:
                sample["metadata"] = r.metadata

            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.records)


class TextStructurer:
    """
    Converts cleaned but unstructured CV / jobdesc into
    model-ready structured text.
    """

    def resume_from_json(self, data: dict) -> str:

        output = []

        output.append("[RESUME]")

        if data.get("summary"):
            output.append(
                f"SUMMARY:\n{data['summary']}"
            )

        skills = data.get("skills", {})

        tech = skills.get("technical", {})

        tech_names = []

        for group in tech.values():
            for item in group:
                tech_names.append(item["name"])

        if tech_names:
            output.append(
                "SKILLS:\n" +
                ", ".join(tech_names)
            )

        for exp in data.get("experience", []):

            output.append(
                f"""
            EXPERIENCE:
            TITLE: {exp.get('title', '')}
            LEVEL: {exp.get('level', '')}

            RESPONSIBILITIES:
            {" ".join(exp.get('responsibilities', []))}
            """
                )

        for edu in data.get("education", []):

            degree = edu.get("degree", {})

            output.append(
                f"""
            EDUCATION:
            {degree.get('level', '')}
            {degree.get('field', '')}
            """
            )

        return "\n\n".join(output)

    def load_jsonl(self, path):
        samples = []

        with open(path, "r", encoding="utf-8") as entry:
            for line in entry:
                item = json.loads(line)
                samples.append(item)
            return samples

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
        Convert messy CV into SAME schema as structure_resume_from_json
        """

        sections = self._split_sections(text)

        output = []
        output.append("[RESUME]")

        # --------------------------
        # TITLE (same logic as JSON version)
        # --------------------------
        title = sections.get("title", [])
        if title:
            output.append(f"TITLE:\n{self._join('TITLE', title)}")

        # --------------------------
        # SUMMARY (new: keep consistent even if heuristic)
        # --------------------------
        summary = sections.get("summary", [])
        if summary:
            output.append(f"SUMMARY:\n{self._join('SUMMARY', summary)}")

        # --------------------------
        # SKILLS (match JSON behavior)
        # --------------------------
        skills = sections.get("skills", [])
        if skills:
            output.append(f"SKILLS:\n{self._join('SKILLS', skills)}")

        # --------------------------
        # EXPERIENCE (structured blocks like JSON version)
        # --------------------------
        exp_lines = sections.get("experience", [])
        if exp_lines:
            output.append("EXPERIENCE:\n" + self._join("EXPERIENCE", exp_lines))

        # --------------------------
        # EDUCATION
        # --------------------------
        edu_lines = sections.get("education", [])
        if edu_lines:
            output.append("EDUCATION:\n" + self._join("EDUCATION", edu_lines))

        return "\n\n".join(output).strip()

    def structure_resume_from_json(self, data: dict) -> str:
        """
        Converts structured resume JSON into unified training schema:
        TITLE / SUMMARY / SKILLS / EXPERIENCE / EDUCATION
        """

        sections = []
        sections.append("[RESUME]")

        # --------------------------
        # SUMMARY
        # --------------------------
        summary = data.get("summary", "")
        if summary:
            sections.append(f"SUMMARY:\n{summary}")

        # --------------------------
        # SKILLS
        # --------------------------
        skills = data.get("skills", {})
        tech = skills.get("technical", {})

        tech_names = []
        for group in tech.values():
            for item in group:
                if isinstance(item, dict) and "name" in item:
                    tech_names.append(item["name"])

        if tech_names:
            sections.append("SKILLS:\n" + ", ".join(tech_names))

        # --------------------------
        # TITLE (IMPORTANT FIX)
        # --------------------------
        title = None
        experiences = data.get("experience", [])

        # primary: first valid experience title
        for exp in experiences:
            if exp.get("title"):
                title = exp["title"]
                break

        if title:
            sections.insert(1, f"TITLE:\n{title}")

        # --------------------------
        # EXPERIENCE (clean structured)
        # --------------------------
        exp_blocks = []

        for exp in experiences:
            block = []

            t = exp.get("title")
            if t:
                block.append(f"TITLE: {t}")

            dates = exp.get("dates", {})
            if isinstance(dates, dict):
                start = dates.get("start")
                end = dates.get("end")
                if start or end:
                    block.append(f"DATES: {start or ''} - {end or ''}")

            if block:
                exp_blocks.append("\n".join(block))

        if exp_blocks:
            sections.append("EXPERIENCE:\n" + "\n\n".join(exp_blocks))

        # --------------------------
        # EDUCATION
        # --------------------------
        edu_blocks = []

        for edu in data.get("education", []):
            degree = edu.get("degree", {})
            institution = edu.get("institution", {})
            dates = edu.get("dates", {})

            parts = []

            level = degree.get("level")
            field = degree.get("field")
            if level or field:
                parts.append(" ".join([level or "", field or ""]).strip())

            if institution.get("name"):
                parts.append(institution["name"])

            start = dates.get("start")
            end = dates.get("expected_graduation") or dates.get("end")
            if start or end:
                parts.append(f"{start or ''}-{end or ''}")

            if parts:
                edu_blocks.append(" ".join(parts))

        if edu_blocks:
            sections.append("EDUCATION:\n" + "\n".join(edu_blocks))

        return "\n\n".join(sections)

    def create_resume_match_score(self, job, resume):

        score = 0

        job_words = set(job.lower().split())

        title = resume.get("TITLE", "")
        skills = resume.get("SKILLS", "")
        experience = resume.get("EXPERIENCE", "")

        title_words = set(title.lower().split())
        skill_words = set(skills.lower().split())
        exp_words = set(experience.lower().split())

        # title is strongest signal
        score += (
            len(job_words & title_words)
            / max(len(job_words), 1)
        ) * 0.2

        # skills
        score += (
            len(job_words & skill_words)
            / max(len(job_words), 1)
        ) * 0.4

        # experience
        score += (
            len(job_words & exp_words)
            / max(len(job_words), 1)
        ) * 0.2

        return min(score, 1.0)

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
# contoh penggunaan dalam training

from dataset import build_dataloader
from preprocess import TextStructurer
from assembler import RerankerAssembler  # assuming split file

processor = TextStructurer()
assembler = RerankerAssembler()

# -------------------------
# 1. JSON → structured document
# -------------------------
resume_json = {
    "summary": "Backend engineer with Python experience",
    "experience": [
        {
            "title": "Python Developer",
            "level": "junior",
            "company": "ABC Corp",
            "responsibilities": ["API development", "DB optimization"],
            "dates": {"start": "2021", "end": "2024"}
        }
    ],
    "skills": {
        "technical": {
            "programming": [
                {"name": "Python"},
                {"name": "FastAPI"}
            ]
        }
    },
    "education": [
        {
            "degree": {"level": "BSc", "field": "Computer Science"},
            "institution": {"name": "XYZ University"}
        }
    ]
}

document_text = processor.structure_resume_from_json(resume_json)

# -------------------------
# 2. Add training sample
# -------------------------
assembler.add_sample(
    query="Looking for a Python backend engineer with API experience",
    document=document_text,
    label=1
)

assembler.add_sample(
    query="Looking for a Python backend engineer with API experience",
    document="Unrelated accountant with finance background",
    label=0
)

# -------------------------
# 3. Export dataset
# -------------------------
samples = assembler.to_samples()

print(samples[0].keys())

# -------------------------
# 4. DataLoader
# -------------------------
loader = build_dataloader(
    samples=samples,
    batch_size=2,
    shuffle=True
)

for batch in loader:
    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)

    break

# contoh menggunakan text

from preprocess import TextStructurer
from dataset import RerankerAssembler
import numpy as np

# init
structurer = TextStructurer()
assembler = RerankerAssembler()

# -------------------------
# read raw text file
# -------------------------
with open("resume.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# -------------------------
# convert raw text → structured resume
# -------------------------
structured_resume = structurer.structure_resume(raw_text)

# -------------------------
# create training pair
# -------------------------
assembler.add_sample(
    query="Looking for a marketing manager with product launch experience",
    document=structured_resume,
    label=1
)

# -------------------------
# export dataset
# -------------------------
samples = assembler.to_samples()

print(samples[0]["document"])
print(samples[0].keys())
'''
