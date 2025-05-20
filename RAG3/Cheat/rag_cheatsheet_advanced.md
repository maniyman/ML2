
# 🧠 RAG Cheat Sheet – Advanced Concepts & Best Practices

## 🔍 Zentrale Konzepte & Begriffe

### 🔸 Cosine Similarity
- Misst den Winkel zwischen zwei Vektoren im Vektorraum.
- Wertebereich: `[-1, 1]` – je näher an `1`, desto ähnlicher.
- Wichtig für die Ähnlichkeitsbewertung bei Embeddings.

### 🔸 Vector Stores
- Speichern unstrukturierte Daten als Vektoren.
- Retrieval erfolgt über Ähnlichkeitsmetriken (z. B. Cosine).
- Beispiele: FAISS, Chroma, Weaviate, Pinecone.

### 🔸 Indexierung in Vector Stores
- Einsatz von **ANN (Approximate Nearest Neighbor)** zur Effizienzsteigerung.
- ANN-Techniken:
  - FAISS (Clustering)
  - HNSW (Graph-basiert)
  - ANNOY (Tree-basiert)
  - LSH (Hashing)
  - PQ (Quantisierung)

### 🔸 Hybrid Search
- Kombination aus Vektor-Suche (semantisch) und Keyword-Suche (symbolisch).
- Verbessert Relevanz durch parallele Strategien.

### 🔸 Distance Thresholding
- Legt maximale Distanz zwischen Query und Dokument fest.
- Nutzt Kombination mit Top-k Retrieval.

### 🔸 Prompt Design
- Strukturiere Prompts mit Tags wie `<information>` für klare Trennung.
- Position der Informationen im Prompt (Anfang/Ende) beeinflusst Gewichtung durch das LLM.

---

## 🛠️ Tools, Libraries & Architekturen

| Kategorie          | Beispiele / Hinweise                              |
|--------------------|----------------------------------------------------|
| Vektor-Datenbanken | FAISS, Chroma, Pinecone, Weaviate                  |
| Frameworks         | LangChain, LlamaIndex, Haystack                    |
| Embeddings         | OpenAI, HuggingFace, SentenceTransformers         |
| Evaluation         | GPT-4 für automatische Antwortbewertung           |

---

## 🧪 Query Transformation Techniken

| Methode                       | Zweck                                            |
|------------------------------|--------------------------------------------------|
| Query Expansion              | Füge hypothetische Antwort in die Query ein     |
| Query Augmentation / Rewrite| Lass LLM alternative, bessere Queries generieren |
| HyDe                         | Erzeuge hypothetische Dokumente zur Retrieval-Verbesserung |
| Subquestions / Planning      | Teile komplexe Fragen in Subfragen auf          |
| Multi-Step Reasoning         | Iterative Subfrage-Generierung durch das LLM    |

---

## 🧠 Lessons Learned & typische Fehler

- ❌ *Falsche Ähnlichkeitsmetrik*: nicht jede Aufgabe passt zu Cosine Similarity.
- ❌ *Zu niedrige Top-k Werte*: relevante Kontexte können übersehen werden.
- ❌ *Unklare Promptstruktur*: beeinträchtigt Modellverhalten.
- ✅ *Query Transformation steigert Recall signifikant*.
- ✅ *Reranking via Cross-Encoder verbessert Precision*.

---

## 🧾 Wichtige Codeblöcke aus dem Notebook `advanced_rag.ipynb`

```python
import tqdm
import glob
from PyPDF2 import PdfReader
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
```

```python
### load the pdf from the path
glob_path = "data/**/*.pdf"
pdf_paths = glob.glob(glob_path, recursive=True)
pdf_texts = []

for path in pdf_paths:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    pdf_texts.append(text)
```

```python
# Create a splitter: 2000 characters per chunk
chunks = []
for text in pdf_texts:
    for i in range(0, len(text), 2000):
        chunk = text[i:i+2000]
        chunks.append(chunk)
```

```python
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
embeddings = model.encode(chunks)
```

```python
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

query = "What are the side effects of the drug?"
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=5)
retrieved_chunks = [chunks[i] for i in I[0]]
```

---

## 🛠 Hinweise zur praktischen RAG-Umsetzung

- **Datenvorverarbeitung**:
  - Chunking (z. B. nach Absätzen oder Tokens)
  - Metadaten anreichern (z. B. Quelle, Zeitstempel)

- **Embedding-Strategie**:
  - Einheitliche Modelle für Query & Dokumente verwenden
  - Spezialisierte Embedding-Modelle für Fachdomänen nutzen

- **Retriever-Tuning**:
  - Top-k und Schwellenwerte experimentell bestimmen
  - Hybrid & MMR-Strategien testen
  - Optional: Benutzerfeedback zur Relevanz nutzen

---

## 📌 Quellen & Referenzen

- [Weaviate Blog: Why Vector Search is Fast](https://weaviate.io/blog/why-is-vector-search-so-fast)
- [Pinecone Tutorials](https://www.pinecone.io/learn/)
- [Towards Data Science RAG Series](https://towardsdatascience.com/tagged/rag)
