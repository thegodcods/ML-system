# hello disini menerima jobdesc, resume, dan mengembalikan ranking score

import torch
from quickhire_model import IndoBERTRanker
from transformers import AutoTokenizer
from global_ml_sys_config import MODEL_NAME, DEVICE

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# define model di awal
model = IndoBERTRanker(
    model_name=MODEL_NAME,
    vector_dim=128,
    hidden_dim=256,
)

# disini melakukan load model weights dalam format .pt
# menyalakan mode eval untuk mengubah perilaku layer
# contohnya adalah mematikan layer droput
# dan menggunakan running stats pada layer batch norm
model.load_state_dict(torch.load("quickhire_reranker.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# retrieval


def retrieve(query: str, candidates: list[str], top_k: int = 10):
    """
    Mengembalikan top 10 hasil dari inference
    kemungkinan bisa dikembangkan dengan FAISS/BM2/EMBEDDING SEARCH
    """
    return candidates[:top_k]


def build_extra_vectors(query: str, docs: list[str]) -> torch.Tensor:
    """
    Placeholder for external text embeddings / features.

    In real systems, this could be:
    - SentenceTransformer embeddings
    - TF-IDF features
    - metadata features
    - click signals
    """
    # mengembalikan vector sebanyak len(docs) dan 128 atau lebih fitur
    return torch.randn(len(docs), 128)


# RERANKING STAGE


def rerank(query: str, candidates: list[str]):
    """
    reranking Cross Encoder berbasis Indobert dan Perceptron sederhana
    """

    # membuat pair dokumen dan kandidat
    queries = [query] * len(candidates)

    enc = tokenizer(
        queries,
        candidates,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    # memanggil torch tanpa menggunakan autograd untuk prediksi
    # autograd hanya khusus training
    # menggunakan model dalam variabel scores
    with torch.no_grad():
        scores = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )

    scores = scores.cpu().tolist()

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# FULL PIPELINE

def infer(request: dict):
    """
    Main inference entry point

    request format:
    {
        "query": str,
        "candidates": list[str]
    }
    """

    query = request["query"]
    candidates = request["candidates"]

    # 1. retrieval top k 10 candidate dengan cepat
    retrieved = retrieve(query, candidates, top_k=10)

    # 2. rerank pemanggilan model
    ranked = rerank(query, retrieved)

    # 3. format output
    return {
        "query": query,
        "results": [
            {
                "text": text,
                "score": float(score)
            }
            for text, score in ranked
        ]
    }


# CLI TEST

"""
if __name__ == "__main__":

    sample_request = {
        "query": "laptop murah untuk mahasiswa",
        "candidates": [
            "Laptop gaming RTX 4090 mahal",
            "Laptop ringan baterai awet untuk kuliah",
            "Mouse wireless murah",
            "Laptop second murah kondisi bagus",
            "Smartphone Android terbaru"
        ]
    }

    result = infer(sample_request)

    print("\nRANKED RESULTS:\n")
    for r in result["results"]:
        print(f"{r['score']:.4f} - {r['text']}")

"""
