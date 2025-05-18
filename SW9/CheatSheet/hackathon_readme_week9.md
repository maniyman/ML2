
# 🧠 Hackathon README – ML2 Woche 9: Transfer Learning & Vision Transformers

Dieses Dokument enthält alle wesentlichen Schritte, um ein Bildklassifikationsmodell mit einem vortrainierten Vision Transformer (ViT) in PyTorch Lightning zu bauen, anzupassen und zu evaluieren. Es eignet sich zur direkten Verwendung im Hackathon.

---

## ✅ Ziel
- Verwendung eines vortrainierten Vision-Transformers (ViT)
- Anpassung auf neue Klassen (z. B. Cats vs Dogs oder Genki4k)
- Training mit PyTorch Lightning
- Evaluation inkl. Confusion Matrix
- Möglichkeit, Modelle gezielt „einzufrieren“ (Feature Extraction)

---

## 🧰 1. Installation und Setup in Google Colab
Führe dies in einer Zelle aus:
```bash
!pip install lightning transformers torchvision scikit-learn --quiet
!wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
!unzip -q cats_and_dogs_filtered.zip
```

Damit wird der Datensatz `cats_and_dogs_filtered` heruntergeladen und entpackt.

---

## 📦 2. Bibliotheken importieren
```python
import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
```

---

## 🖼️ 3. Daten vorbereiten
```python
# FeatureExtractor für ViT laden
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Transformation: Resize, Tensor, Normalisieren
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3] if x.shape[0] == 3 else x.repeat(3,1,1)),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Pfade
train_dir = "/content/cats_and_dogs_filtered/train"
val_dir = "/content/cats_and_dogs_filtered/validation"

# Dataset & Dataloader
train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)
```

---

## 🔁 4. Optional: Vortrainiertes Modell einfrieren (Feature Extraction)
```python
# Beispiel: Alle Gewichte einfrieren (außer Klassifikator)
def freeze_all_except_classifier(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
```

---

## ⚡ 5. LightningModule definieren (Vision Transformer)
```python
class ViTLightningModule(L.LightningModule):
    def __init__(self, num_labels):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        # Optional: Feature Extraction aktivieren
        # freeze_all_except_classifier(self.model)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(pixel_values=x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs.logits, y)
        acc = (outputs.logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        val_loss = self.loss_fn(outputs.logits, y)
        val_acc = (outputs.logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)
```

---

## 🚀 6. Training mit PyTorch Lightning starten
```python
model = ViTLightningModule(num_labels=2)
trainer = L.Trainer(max_epochs=3, accelerator="auto", devices="auto")
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

---

## 📊 7. Evaluation: Accuracy & Confusion Matrix
```python
model.eval()
preds = []
labels = []

for batch in val_loader:
    x, y = batch
    with torch.no_grad():
        out = model(x.to(model.device))
    preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
    labels.extend(y.numpy())

# Accuracy
acc = accuracy_score(labels, preds)
print(f"Validation Accuracy: {acc:.2%}")

# Confusion Matrix anzeigen
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_ds.classes)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
```

---

## 📌 Hinweise für den Hackathon
- ✅ Aktiviere GPU unter `Laufzeit > Laufzeittyp ändern > GPU`
- ✅ Daten müssen in `ImageFolder`-Struktur vorliegen
- ✅ RGB-Bilder: ein Kanalbild muss auf 3 Kanäle erweitert werden (`x.repeat(3, 1, 1)`)
- ✅ Für andere Modelle (DINOv2 etc.): Ersetze `ViTForImageClassification`

---

Viel Erfolg beim Hackathon! 💪
