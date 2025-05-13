
# 🧠 Erkenntnisse & Fehleranalyse – SW8: CNNs mit CIFAR10 (ML2 FS2025)

Dies ist eine persönliche Zusammenfassung aller Übungen und Erkenntnisse, die ich während meiner Arbeit mit Woche 8 gemacht habe. Der Fokus liegt auf den **Fehlern**, die ich gemacht habe, und dem, was ich daraus gelernt habe.

---

## ✅ Überblick: Was ich gemacht habe

| Übung | Thema | Ziel erreicht? |
|-------|-------|----------------|
| 1     | MNIST-Modell auf CIFAR10 anwenden | ✅ Ja |
| 2     | Erstes CNN für Bilder bauen | ✅ Ja |
| 3     | Regularisierung einbauen (Dropout, L2, EarlyStopping) | ✅ Ja |
| 4     | TensorBoard zur Visualisierung verwenden | ✅ Ja |

---

## ❌ Fehler & Was ich daraus gelernt habe

### 🔻 Fehler 1: `Conv2D` nach `Flatten()`
**Was passiert ist:**  
Ich habe einen `Conv2D`-Layer NACH einem `Flatten()` eingefügt.

**Warum das falsch war:**  
`Flatten()` macht aus einem Bild ein 1D-Vektor. `Conv2D` braucht aber ein 4D-Format (Batch, Höhe, Breite, Kanäle).

**Richtig:**  
Conv-Schichten gehören **immer vor** Flatten.

---

### 🔻 Fehler 2: Optimizer als String übergeben
```python
optimizer='Adam'  # FALSCH
```

**Warum das falsch war:**  
Ich habe `'Adam'` als String übergeben, obwohl ich vorher `Adam = tf.keras.optimizers.Adam(...)` definiert habe.

**Richtig:**
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, ...)
```

---

### 🔻 Fehler 3: Lernrate zu hoch
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)  # VIEL ZU HOCH
```

**Warum das schlecht war:**  
Adam arbeitet intern mit adaptiven Lernraten. `0.1` führt fast immer zu schlechtem oder instabilem Training.

**Empfohlen:**  
```python
learning_rate = 0.001
```

---

### 🔻 Fehler 4: TensorBoard-Callback statisch
```python
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs')
```

**Warum nicht optimal:**  
Alle Trainingsläufe landen im gleichen Ordner – keine gute Vergleichbarkeit.

**Besser:**
```python
from datetime import datetime
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
```

---

## 🧱 Richtige Reihenfolge von Layern in einem CNN

Eine bewährte Reihenfolge für CNN-Modelle ist:

```
Input (z. B. 32x32x3 Bild)
    ↓
Conv2D (z. B. 32 Filter, 3x3)
    ↓
Activation (z. B. ReLU)
    ↓
MaxPooling2D (z. B. 2x2)
    ↓
[Mehr Conv + Pooling-Schichten]
    ↓
Flatten()
    ↓
Dense (z. B. 64 Neuronen)
    ↓
Dropout (optional)
    ↓
Dense (Output, z. B. 10 Klassen mit Softmax)
```

📌 Wichtig:
- `Conv2D` und `MaxPooling2D` kommen **vor** `Flatten()`
- `Flatten()` bereitet die Daten für `Dense` vor
- Dropout vor dem finalen Klassifikator hilft gegen Overfitting

---

## 🎯 Was ich jetzt besser kann

- CNNs bauen und verstehen (Conv → Pool → Flatten → Dense)
- Mit Bilddaten arbeiten (RGB, Shapes, Normalisierung)
- Fehler analysieren und gezielt beheben
- Regularisierung richtig einsetzen
- Trainingsverlauf mit TensorBoard überwachen

---

**Ende der SW8-Zusammenfassung**  
Erstellt am: automatisch mit ChatGPT (Mai 2025)
