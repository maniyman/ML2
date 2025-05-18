# 🧠 RAG Pipeline Cheat Sheet – PDF → FAISS → Querying

Dieses Cheat Sheet beschreibt eine vollständige RAG-Pipeline, die PDF-Dokumente verarbeitet, in Chunks unterteilt, als semantische Vektoren kodiert und in einem FAISS-Index speichert. Die Chunks können später über eine Benutzereingabe abgerufen und visualisiert werden. Die Lösung ist modular, flexibel und bereit für produktionsnahe Anwendungen.

---

## ✅ Step 1: Setup – Import & Installation

Bevor wir mit der Verarbeitung beginnen, installieren wir benötigte Pakete wie `PyPDF2` (PDF-Verarbeitung), `langchain-community`, `faiss-cpu` (für schnellen Ähnlichkeitsvergleich) und laden alle relevanten Python-Module. Dies stellt sicher, dass alle notwendigen Tools zur Verfügung stehen.

```python
!pip install pypdf2 langchain-community faiss-cpu

import tqdm, glob
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss, pickle
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np
```

---

## ✅ Step 2: Daten aus PDFs extrahieren

In diesem Schritt laden wir alle PDF-Dateien aus einem Verzeichnis und extrahieren deren Textinhalte. Diese Texte bilden später unsere Wissensquelle. Es wird sichergestellt, dass nur nicht-leere Seiten berücksichtigt werden, um Fehler zu vermeiden.

```python
glob_path = "data/*.pdf"
text = ""
for pdf_path in tqdm.tqdm(glob.glob(glob_path)):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text += " ".join(page.extract_text() for page in reader.pages if page.extract_text())
```

---

## ✅ Step 3: Text in überlappende Chunks splitten

Lange Texte sind schwer zu verarbeiten und passen oft nicht in ein LLM-Fenster. Wir teilen sie deshalb in kleinere, überlappende Chunks, um den Kontext zu erhalten. Das hilft später beim Einbetten und Abrufen der Informationen.

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)
```

---

## ✅ Step 4: Token-basiertes Splitten (optional)

Um besser auf Token-Limits der Sprachmodelle Rücksicht zu nehmen, wird zusätzlich ein tokenizerbasierter Split durchgeführt. Das ist besonders hilfreich bei multilingualen Daten oder bei der Vorbereitung für das Embedding.

```python
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=128, model_name="paraphrase-multilingual-MiniLM-L12-v2")
token_split_texts = []
for text in chunks:
    token_split_texts += token_splitter.split_text(text)
```

---

## ✅ Step 5: Embeddings für Text-Chunks erzeugen

Jetzt erzeugen wir semantische Embeddings für jeden Chunk mit einem SentenceTransformer-Modell. Die Embeddings sind dichte Vektoren, die die Bedeutung des Textes kodieren. Sie bilden die Grundlage für unsere spätere semantische Suche.

```python
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chunk_embeddings = model.encode(token_split_texts, convert_to_numpy=True)
```

---

## ✅ Step 6: FAISS-Vektorindex erstellen und speichern

FAISS ist eine Bibliothek für schnelle Ähnlichkeitssuche. Wir speichern alle Embeddings im FAISS-Index und sichern zusätzlich die Text-Chunks separat als Mapping, damit wir später passende Texte zu Vektoren zurückführen können.

```python
d = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(chunk_embeddings)

faiss.write_index(index, "faiss/faiss_index.index")
with open("faiss/chunks_mapping.pkl", "wb") as f:
    pickle.dump(token_split_texts, f)
```

---

## ✅ Step 7: Embeddings mit UMAP in 2D projizieren

Um einen visuellen Überblick über die semantischen Zusammenhänge der Daten zu bekommen, projizieren wir die hochdimensionalen Embeddings mit UMAP in einen zweidimensionalen Raum. Dies hilft, Cluster und Themen zu erkennen und Retrieval-Qualität zu evaluieren.

```python
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(chunk_embeddings)

def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm.tqdm(embeddings, desc="Projecting Embeddings")):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

projected_dataset_embeddings = project_embeddings(chunk_embeddings, umap_transform)
```

---

## ✅ Step 8: Semantische Suche (Retrieval)

Jetzt definieren wir eine Funktion, die eine Benutzereingabe (Query) entgegennimmt, diese einbettet und im FAISS-Index nach den ähnlichsten Chunks sucht. Die gefundenen Ergebnisse können optional visualisiert werden.

```python
def retrieve(query, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_texts = [token_split_texts[i] for i in indices[0]]
    retrieved_embeddings = np.array([chunk_embeddings[i] for i in indices[0]])
    return retrieved_texts, retrieved_embeddings, distances[0]

query = "machine learning algorithms"
results, result_embeddings, distances = retrieve(query, k=3)
```

---

## ✅ Step 9: Visualisierung der Abfrage und Ergebnisse

Wir projizieren die Query, die abgerufenen Ergebnisse und das gesamte Dataset gemeinsam in den 2D-Raum. So können wir nachvollziehen, ob das Retrieval semantisch sinnvolle Ergebnisse liefert.

```python
projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)
query_embedding = model.encode([query], convert_to_numpy=True)
project_original_query = project_embeddings(query_embedding, umap_transform)

def shorten_text(text, max_length=15):
    return (text[:max_length] + '...') if len(text) > max_length else text

plt.figure()
plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray', label='Dataset')
plt.scatter(projected_result_embeddings[:, 0], projected_result_embeddings[:, 1], s=100, facecolors='none', edgecolors='g', label='Results')
plt.scatter(project_original_query[:, 0], project_original_query[:, 1], s=150, marker='X', color='r', label='Original Query')

for i, text in enumerate(results):
    if i < len(projected_result_embeddings):
        plt.annotate(shorten_text(text), (projected_result_embeddings[i, 0], projected_result_embeddings[i, 1]), fontsize=8)

plt.annotate(shorten_text(query), (project_original_query[0, 0], project_original_query[0, 1]), fontsize=8)
plt.gca().set_aspect('equal', 'datalim')
plt.title('RAG Embedding Visualisation')
plt.legend()
plt.show()
```

---

## ✅ Ergebnis

Du hast eine vollständige, robuste und erweiterbare RAG-Pipeline gebaut – vom Dokument bis zur semantischen Suche mit Visualisierung und LLM-Anbindung.