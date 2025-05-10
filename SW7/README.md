# 📚 ML2 Week 7 Cheat Sheet: Artificial Neural Networks (ANNs)

Dieses Repository enthält eine umfassende Cheat Sheet-Zusammenfassung der Inhalte aus Woche 7 des Machine Learning II Kurses (FS 2025) an der ZHAW.  
Der Fokus liegt auf Artificial Neural Networks (ANNs), praktischen Aufgabenmustern und allen wichtigen Themen, die für die Vorbereitung auf einen hackathon-artigen Prüfungstag relevant sind.

## 🏗 Inhalt

✅ Kurze Definitionen der Schlüsselkonzepte  
✅ Typische Aufgabenmuster (z. B. Fine-Tuning, Trainingsschleifen, Inferenz, Evaluierung)  
✅ Python-Codebeispiele mit ausführlichen Kommentaren  
✅ Relevante Libraries & Frameworks  
✅ Best Practices & Troubleshooting Tipps  
✅ Empfohlene Hyperparameter/Setups  
✅ Checkliste „Hast du alles überprüft?“

---

## 📦 Dateien

| Datei                   | Beschreibung                                      |
|-------------------------|--------------------------------------------------|
| `cheat_sheet.md`        | Die umfassende Cheat Sheet-Zusammenfassung       |
| `notebooks/`           | Übungs-Notebooks aus Woche 7 (EMNIST, MNIST etc.)|
| `slides/W7_ANN.pdf`    | Originale Kurs-Slides zu Woche 7                 |

---

## 🛠 Verwendete Tools & Frameworks

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [PyTorch (optional)](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Keras Tuner](https://keras.io/keras_tuner/)

---

## 🚀 Setup & Ausführung

1️⃣ **Environment vorbereiten:**  
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2️⃣ **Notebook starten (optional):**  
```bash
jupyter notebook
```

3️⃣ **TensorFlow testen:**  
```python
import tensorflow as tf
print(tf.__version__)
```

---

## 💡 Hinweise & Best Practices

- Stelle sicher, dass du CUDA/GPU aktiviert hast, wenn du große Modelle trainierst.
- Setze Zufallsseeds (`tf.random.set_seed(42)`), um reproduzierbare Ergebnisse zu erhalten.
- Überprüfe immer die Input/Output-Dimensionen deiner Daten.
- Nutze Early Stopping und Model Checkpoints, um Overfitting zu vermeiden.

---

## ✅ Checkliste vor der Prüfung

- [ ] Läuft der Code lokal oder in Google Colab?
- [ ] Sind alle Daten und Modelle sauber vorbereitet?
- [ ] Hast du die Hyperparameter abgestimmt?
- [ ] Sind alle relevanten Libraries installiert?
- [ ] Hast du das Training und die Evaluation getestet?

---

## 📄 Lizenz

Dieses Projekt dient ausschließlich Lern- und Übungszwecken im Rahmen des Machine Learning II Kurses (FS 2025, ZHAW).

---

## 👩‍🏫 Kontakt

Dr. Elena Gavagnin  
Machine Learning II, ZHAW  
elena.gavagnin@zhaw.ch
