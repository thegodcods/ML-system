# Cara menggunakan model untuk inference

## 1. instalasi dependency

pembangunan env menggunakan python 3.13.3 

beware training model reranker menggunakan pytorch 2.5.1 dengan torchvision 0.20.1 dan torchaudio 2.5.1

```bash
pip install -r requirements.txt

```

## 2. Script dan hal yang dibutuhkan untuk inference

1. quickhire_model.py untuk loading architecture model
2. preprocess.py untuk preprocessing otomatis raw text query dan raw text document
3. weights.pt sebagai logic dari sistem inference
4. inference.py sebagai gateway utama sistem backend dan sistem ML
5. global_ml_sys_config.py sebagai file config central

## 3. cara kerja script inference: 

section berikut
```
structurer = TextStructurer()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# define model di awal
model = IndoBERTRanker(
    model_name=MODEL_NAME,
    hidden_dim=256
)
```
memanggil model transformer, class preprocessing text beserta fungsi dalam kelas dan memanggil arsitektur model dalam `quickhire_model`

kemudian dalam inference, weights training digunakan pada baris berikut
```
model.load_state_dict(torch.load("quickhire_reranker.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()
```
jangan lupa untuk menyalakan eval mode agar training tidak terjadi tengah inference

dan setelah ini ada function `infer()` sebagai gateway utama dalam mengirim request inference

silahkan ganti best.pt yang telah ditraining sebelumnya ke nama quickhire_reranker.pt agar bisa dideteksi script infer dengan benar


## 4. Melakukan inference

1. backend mengirim request dari berbentuk query dan dokumen cv dalam bentuk berikut:

```
{
 query:"Looking for Python backend engineer",
 candidates:[
    resume1,
    resume2,
    resume3
 ]
}
```

2. request dikirim ke function infer

```
from infer import infer

request = {
    "query": "dicari data scientist untuk membangun modek prediksi dan analisis data menggunakan python, sql, machine learning, pandas numpy",
    "candidates": [
        """
        Skills:
        Python
        FastAPI
        PostgreSQL

        Experience:
        Backend Engineer
        """,

        """
        Skills:
        React
        HTML
        CSS

        Experience:
        Frontend Developer
        """
    ]
}

result = infer(request)

print("\nRANKED RESULTS:\n")
    for r in result["results"]:
        print(f"{r['score']:.4f} - {r['text']}")
```

Note: 
* infer dilakukan melalui function `infer()`. ini dikarenakan infer sudah memanggil function `rerank()` yang memproses data dari raw text ke format yang dikenali model 
* fungsi `rerank` memanggil Class `TextStructurer` dari script `preprocess` yang memiliki fungsi `structure_resume()` yang menerima teks document dan mengekstrak kolom title, summary, skills, experience dan education
* model menerima `list[str]` pada function `infer()` untuk batch inference