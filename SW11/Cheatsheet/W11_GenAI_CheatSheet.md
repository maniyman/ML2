
# README – Week 11: Generative AI (ML2 FS2025)

Dieses Repository enthält eine umfassende Markdown-Cheat-Sheet-Datei zur Vorbereitung auf die Hackathon-Prüfung im Rahmen des Machine Learning II Kurses (FS 2025, Woche 11). Das Hauptthema dieser Woche ist "Generative AI" mit Fokus auf Autoencoder, Variational Autoencoder (VAE), Generative Adversarial Networks (GANs) und Diffusionsmodelle wie Stable Diffusion.

## Inhalt
- `W11_GenAI_CheatSheet.md`: Die zentrale Lern- und Referenzdatei mit:
  - Definitionen & Konzepte
  - Codebeispielen (kommentiert)
  - Typische Aufgabenstellungen (aus Übungen)
  - Best Practices & Troubleshooting
  - Finaler Vorbereitungs-Checkliste

## Voraussetzungen
- Python 3.8+
- GPU empfohlen (z. B. mit CUDA-Unterstützung)
- Bibliotheken:
  - TensorFlow, Keras
  - PyTorch (optional für Diffusion)
  - Hugging Face `transformers`, `diffusers`
  - NumPy, OpenCV, matplotlib, PIL

## Quickstart (Stable Diffusion)
```bash
pip install diffusers transformers accelerate torch
```
```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
pipe("fantasy castle at sunset").images[0].show()
```

## Lizenz
Die Inhalte basieren auf den Unterlagen von Dr. Elena Gavagnin, ZHAW.

---

# W11 Generative AI Cheat Sheet

## 🌍 Kernkonzepte

### Autoencoder (AE)
- Lernen komprimierter (latenter) Darstellungen der Eingabe.
- Architektur: Encoder → Bottleneck → Decoder.
- Anwendung: Denoising, Pretraining, DimRed, Generator.

### Variational Autoencoder (VAE)
- Lernen Verteilungen in latenten Räumen → Sampling für Bildgenerierung.
- Encoder gibt Mittelwert & Varianz zurück → Sampling aus Gauss.

### GANs
- Generator + Discriminator: Wettbewerb zur Bildgenerierung.
- Generator lernt durch Feedback des Discriminators.

### Diffusion Models (z. B. Stable Diffusion)
- Generieren Bilder durch iterative Rausch-Entfernung mit U-Net.
- Optional: Text-Guidance über z. B. CLIP-Encoder.

---

## 🔧 Typische Aufgaben (aus Übungen)

### AE Training Loop (Keras)
```python
stacked_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),
])
stacked_decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(28 * 28),
    tf.keras.layers.Reshape([28, 28]),
])
autoencoder = tf.keras.Sequential([stacked_encoder, stacked_decoder])
autoencoder.compile(loss="mse", optimizer="nadam")
autoencoder.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
```

### Stable Diffusion Beispiel (Inference)
```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
img = pipe("wizard cat, fantasy, Pixar").images[0]
img.show()
```

---

## 🚧 Best Practices & Troubleshooting

### Do
- Seeds setzen: `np.random.seed()`, `tf.random.set_seed()`
- `to('cuda')` nicht vergessen bei Inferenz
- Validierungsdaten benutzen
- Modell speichern

### Don’t
- Batch-Größe zu groß auf GPU
- Vergessen: Discriminator einfrieren beim GAN-Training
- Textprompt falsch formulieren (zu kurz oder mehrdeutig)

---

## ⚖️ Hyperparameter-Tipps
| Modell | Empfehlungen |
|--------|--------------|
| AE | latent_dim=30-100, Optimizer=Nadam/Adam |
| VAE | KL-Div. Loss, latent sampling |
| GAN | lr=1e-4, Binary Crossentropy |
| Diffusion | steps=25-50, scheduler default lassen |

---

## ✅ Checkliste
- [ ] CUDA/GPU aktiv?
- [ ] Seeds gesetzt?
- [ ] Eingabe-/Ausgabe-Shape konsistent?
- [ ] Normierung der Bilder korrekt?
- [ ] Prompt klar formuliert?
- [ ] Validierung implementiert?
- [ ] Speicherverbrauch im Blick?
- [ ] Modelle gespeichert?
