# 🧠 RAG Part 1 – Cheat Sheet (LangChain + Chroma + Gemini)

## 📌 Überblick: Retrieval-Augmented Generation (RAG)

RAG kombiniert zwei Hauptkomponenten:
1. **Retrieval**: Relevante Textausschnitte aus externen Quellen suchen (z. B. ChromaDB).
2. **Generation**: Antwort mit einem LLM generieren (z. B. Google Gemini), basierend auf den gefundenen Informationen.

---

## 🔧 Komponenten des RAG-Workflows

1. **Text-Splitting** – Zerlegt lange Dokumente in überlappende Chunks
2. **Embedding-Erzeugung** – Wandelt Text in semantische Vektoren um
3. **Vektor-Datenbank** – Speichert Embeddings zur späteren semantischen Suche
4. **Retriever** – Findet passende Textausschnitte zu einer User-Query
5. **LLM + Prompt Injection** – Generiert die Antwort basierend auf gefundenem Kontext

---

## 🧩 Code mit Erklärungen

```python
# Step 1: (Optional) Installation der Libraries
# !pip install langchain langchain-community langchain-google-genai

# Step 2–3: Benötigte Module importieren
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os
```

```python
# Step 5: API Key für Google Gemini setzen (unbedingt echten Key verwenden)
os.environ["GOOGLE_API_KEY"] = "your_api_key"
```

```python
# Step 7: PDF laden
loader = PyPDFLoader("2023_MicroLED_Pharma_EN_vf.pdf")
pages = loader.load()
```

```python
# Step 8: Erste Seite anzeigen (zur Kontrolle)
pages[0]
```

```python
# Step 10: Text in überlappende Chunks aufteilen
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(pages)
```

```python
# Step 11: Anzahl der Chunks anzeigen
len(texts)
```

```python
# Step 13: Embeddings erzeugen und in Chroma speichern
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma.from_documents(texts, embeddings)
```

```python
# Step 15: Retriever erstellen
retriever = db.as_retriever()
```

```python
# Step 17: RetrievalQA-Chain aufbauen (Retriever + LLM + Prompt-Injection)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-pro"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

```python
# Step 19: Frage stellen (Query)
query = "What is microLED and what is it used for?"
result = qa_chain(query)
```

```python
# Step 21: Antwort anzeigen
print(result["result"])

# Optional: Quellen anzeigen
# result["source_documents"]
```

---

## ✅ Lessons Learned

- Text-Chunks mit Overlap bewahren Kontext
- Embeddings machen Texte maschinell „vergleichbar“
- Vektor-Datenbanken sind essenziell für semantisches Retrieval
- Prompt-Injection ersetzt Fine-Tuning
- RAG reduziert Halluzinationen und erhöht Relevanz

---

## 🧠 Typische Fehler

- Kein Overlap beim Chunking → Kontextverlust
- Falsches Embedding-Modell → schlechte Retrieval-Ergebnisse
- chain_type="stuff" bei großen Mengen unpraktisch (Kontextlimit!)
- Fehlender API-Key oder falscher Dateipfad

---

## 🛠 Tools & Libraries

- LangChain (Chain-Management)
- Chroma (Vektor-Datenbank)
- Google Generative AI (Embeddings + Chat)
- PDFLoader & TextSplitter (Preprocessing)

---

## 🔍 Nächste Schritte

- Retrieval-Optimierung (Top-k, Filter, Hybrid-Ansätze)
- Vergleich: `stuff` vs `map_reduce` Chain-Typ
- Erweiterung auf andere Dateiformate
- Anbindung an UI oder API-Endpunkt
