
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
