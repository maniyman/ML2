# 🧠 ML2 Praxiszusammenfassung – Hackathon-Vorbereitung

## 👤 Teilnehmer
**Name:** _(bitte selbst ergänzen)_  
**Kurs:** Machine Learning 2 (FS 2025)  
**Dozentin:** Dr. Elena Gavagnin

---

## ✅ Überblick: Was wurde geübt?

| Übung | Thema |
|-------|-------|
| Ü1 | Klassisches MLP-Modell auf MNIST |
| Ü2 | Hyperparameter-Experimente |
| Ü3 | EMNIST Buchstabenklassifikation |
| Ü4 | Modell speichern & laden |
| Ü5 | Fehleranalyse & Reflexion |
| Mini-Challenge | Fashion MNIST mit Vorhersageprüfung |

---

## 🧪 Übung 1: MNIST Klassifikation

### ✅ Aufgaben
- Laden & Normalisieren von MNIST
- Einfaches MLP mit 2 Dense-Schichten
- Dropout gegen Overfitting
- EarlyStopping als Callback

### 🧩 Codeauszug
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### ⚠️ Anfangsproblem
- `input_shape=(28, 28, 1)` war falsch für MNIST
- `input_dim` unnötig gesetzt → später entfernt

---

## ⚙️ Übung 2: Hyperparameter-Tests

### ✅ Variationen getestet
- Neuronenzahl: 16, 64, 128
- Optimizer: SGD, Adam, RMSprop
- Learning Rate: 0.0001 bis 0.01
- Batch Size: 32 bis 256

### 🧠 Erkenntnisse
- Zu kleine Netze + zu kleine LR = schlechtes Lernen
- Adam + 128 Neuronen + 0.001 LR + 64 Batch = 💡 sehr stabil

---

## 🔤 Übung 3: EMNIST Letters

### ✅ Aufgaben
- Datensatz geladen & Labels von 1–26 auf 0–25 umgewandelt
- Modell mit 256 Neuronen, Dropout, EarlyStopping
- Softmax-Ausgabe mit 26 Klassen

### 🧠 Problem gelöst
- Einzelne Klasse wurde falsch klassifiziert (Q als A)
- Selbst analysiert: Handschrift schwer lesbar, Modell liegt nicht völlig falsch

---

## 💾 Übung 4: Modell speichern & laden

### ✅ Aufgaben
- Modell mit `model.save()` gespeichert
- Mit `load_model()` neu geladen
- Vorhersagen mit `predict()` gemacht

### 📊 Beispielcode
```python
model.save("emnist_model.keras")
loaded_model = tf.keras.models.load_model("emnist_model.keras")
```

---

## 🔎 Übung 5: Fehleranalyse

### 📌 Fallbeispiel
- Modell hat 1 Beispiel falsch klassifiziert (A ↔ Q)
- Reflexion: Handschrift schwer lesbar, akzeptabler Fehler

---

## 👟 Mini-Challenge: Fashion MNIST

### ✅ Aufgaben
- Komplettes Training auf Fashion MNIST
- 5 zufällige Vorhersagen inkl. Bilder
- Vergleich Vorhersage vs. echtes Label mit ✅/❌

### 📸 Visualisierung
```python
for i in indices:
    img = X_test[i]
    label = y_test[i]
    pred = np.argmax(model.predict(img.reshape(1, 28, 28)))
    status = '✅' if pred == label else '❌'
    plt.imshow(img, cmap="gray")
    plt.title(f"Vorhersage: {class_names[pred]}, Wahr: {class_names[label]} {status}")
    plt.axis("off")
    plt.show()
```

---

## 🧠 Reflexion: Wo gab’s Mühe?

| Thema | Herausforderung | Gelöst durch |
|-------|------------------|--------------|
| Input-Shape in Flatten | Klammern zu tief (28, 28, 1) statt (28, 28) | Korrektur nach Hinweis |
| EarlyStopping | Zuerst definiert, aber nicht eingebaut | Rückmeldung erhalten |
| Vorhersageausgabe | Falscher Vergleich (`X_train` statt `y_train`) | Codehilfe genutzt |
| `plt.title()`-Syntax | Falsche Platzierung von `if`-Ausdruck | Korrekte Formatierung gezeigt |

---

## 🏁 Fazit

Du hast alle zentralen ML-Konzepte praktisch angewendet:
- Modellierung, Evaluation, Speicherung
- Vorhersage, Visualisierung, Reflexion
- Fehler erkennen und verstehen

---

## 🧭 Nächste Schritte (geplant)
- [ ] 📄 Mock-Prüfung durchführen
- [ ] 🔁 Wiederholung mit CNN oder Transfer Learning
- [ ] 🚀 Eigenes Mini-Projekt mit realen Daten

---
