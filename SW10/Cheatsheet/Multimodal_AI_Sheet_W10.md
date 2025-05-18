
# 📘 Multimodal AI – Modelle laden, anwenden, interpretieren und kombinieren  
**ML2 – Woche 10: YOLO, Gemini, GPT-4V**

---

## 🧠 Ziel dieses Sheets

Dieses Sheet zeigt dir:
- ✅ Wie du die Modelle **lädst und verwendest**
- ✅ Welche **Tools für welche Aufgaben** geeignet sind
- ✅ Wie du die **Ergebnisse interpretierst und kombinierst**

Kein Training, keine Theorie – nur praxisnahe Anwendung.

---

## 📌 1. YOLO – Objekterkennung mit Bounding Boxes

### ✅ Was ist das?
YOLO erkennt Objekte in Bildern oder Videos und gibt dir **exakte Koordinaten + Labels**.

### 🛠️ So benutzt du YOLO:

```python
# Installiere YOLOv8
!pip install ultralytics

# Modell laden
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Modell auswählen

# Bild analysieren
results = model("DEIN_BILD.jpg")  # <-- Passe den Bildpfad an

# Ergebnisse anzeigen
results[0].show()

# Labels + Konfidenzwerte ausgeben
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    label = results[0].names[cls_id]
    conf = box.conf[0].item()
    print(f"{label}: {conf:.2f}")
```

📌 **Nutzen:**  
- Ideal, wenn du wissen willst **wo genau etwas ist** (z. B. „Tasse bei x=124, y=88“)

---

## 📌 2. Gemini – Bildkontext verstehen (Google)

### ✅ Was ist das?
Gemini erkennt, **was auf einem Bild passiert** (z. B. „Person liegt am Boden“, „etwas sieht gefährlich aus“).

### 🛠️ Anwendung:

Du gibst Gemini ein Bild + Prompt wie:
> „Welche Objekte befinden sich auf dem Tisch? Wer wirkt verletzt? Was passiert in der Szene?“

📌 **Ergebnis:**
Text mit semantischer Interpretation.  
Kein Code nötig – du nutzt Gemini über Web-Interface oder API.

📌 **Nutzen:**
- Ideal, wenn du verstehen willst, **was die Szene bedeutet**
- Kombinierbar mit YOLO: Gemini sagt „wofür“, YOLO sagt „wo“

---

## 📌 3. OpenAI GPT-4 mit Vision – Sprachlich strukturierte Bildanalyse

### ✅ Was ist das?
GPT-4V versteht Bilder und beschreibt sie in Textform – wie Gemini, aber vorsichtiger.

### 🛠️ Anwendung (via Webinterface / API):

> „Dieses Bild zeigt eine Unfallszene. Welche Personen brauchen Hilfe? Beschreibe jede Person.“

📌 **Nutzen:**
- Für strukturierte Entscheidungslogik (z. B. Triage)
- Kein Bounding Box Output, aber gute Beschreibung

---

## 🔀 Kombination der Tools: Was benutze ich wann?

| Aufgabe | Modell |
|--------|--------|
| 🎯 Objekt exakt erkennen | **YOLO** |
| 🧠 Szene beschreiben & deuten | **Gemini / GPT-4V** |
| 🧩 Semantische Auswahl treffen (z. B. 'wichtigstes Objekt') | **GPT / Gemini** |
| 🗣️ Text in Sprache umwandeln | **gTTS / pyttsx3** |
| 🤖 Gesamtsystem aufbauen | **YOLO + GPT/Gemini kombiniert** |

---

## 🧠 Ergebnisse interpretieren

### YOLO:
- Gibt dir `label`, `confidence`, `xyxy` (Koordinaten)
- Du weißt genau, wo das Objekt im Bild ist
- Nutze es für Greifroboter, Tracking, Bounding Box Visualisierung

### Gemini / GPT:
- Gibt dir Text → „Person liegt bewusstlos links am Boden“
- Du weißt, **was das Objekt bedeutet**
- Nutze es für Entscheidungslogik, Sprachassistenz, Kontextverarbeitung

---

## ✅ Best Practice: Kombiniere die Stärken

Beispiel:
- 🧠 Gemini sagt: „Die Tasse neben dem Messer ist gefährlich.“
- 📦 YOLO erkennt: Es gibt 3 Tassen, 1 Messer → du findest die richtige Tasse per Position

---

## 🧩 Fazit

> Nutze YOLO für präzise Koordinaten.  
> Nutze GPT oder Gemini für Bedeutung und Kontext.  
> Kombiniere beide – dann bist du bereit für deinen Hackathon.
