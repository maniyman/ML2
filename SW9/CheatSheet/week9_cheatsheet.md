# 🧠 ML2 Week 9 Cheat Sheet – Transfer Learning & CNNs (FS2025)

## 📌 Key Concepts

### 🔹 Convolutional Neural Networks (CNNs)
- Specialized neural networks for image data.
- Use convolutional layers to extract spatial features.
- Typical pipeline: Conv -> Activation -> Pooling -> Fully Connected.

### 🔹 Transfer Learning
- **Reuse pre-trained networks** (trained on large datasets like ImageNet).
- **Feature Extraction**: Freeze conv base; train new classifier head.
- **Fine-Tuning**: Unfreeze top conv layers + train classifier head.

### 🔹 Famous Architectures
- **LeNet-5 (1998)**: Simple, MNIST.
- **AlexNet (2012)**: ReLU, dropout, large-scale image tasks.
- **GoogLeNet (2014)**: Inception modules, deep with fewer params.
- **ResNet (2015)**: Residual blocks, skip connections.

### 🔹 Data Augmentation
- Artificially increases dataset size for regularization.
- Methods: flip, rotate, crop, scale, brightness/contrast shift.

### 🔹 Vision Transformers (ViTs)
- Transformers adapted for vision, treating images as patch sequences.

## 🧰 Frameworks & Libraries

| Library | Purpose |
|--------|---------|
| `tensorflow.keras` | Model building & training |
| `tensorflow_hub` | Import pre-trained models |
| `huggingface/transformers` | Vision Transformers |
| `opencv` | Image processing |
| `matplotlib` | Visualization |
| `sklearn` | Evaluation metrics |

## 🧪 Typical Task Patterns & Code

### 🟩 Load Pretrained CNN (TF/Keras)
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### 🟨 Fine-Tune Top Layers
```python
for layer in base_model.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### 🟦 Data Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             horizontal_flip=True, zoom_range=0.2, rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory("data/train", target_size=(224, 224), batch_size=32, class_mode='sparse', subset='training')
val_gen = datagen.flow_from_directory("data/train", target_size=(224, 224), batch_size=32, class_mode='sparse', subset='validation')
```

### 🟧 Training
```python
model.fit(train_gen, epochs=10, validation_data=val_gen)
```

### 🟪 Evaluation & Prediction
```python
loss, acc = model.evaluate(test_gen)
import numpy as np
probs = model.predict(test_gen)
preds = np.argmax(probs, axis=1)
```

### 🟫 Vision Transformers (Hugging Face)
```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=10)
```

## ⚠️ Best Practices & Common Pitfalls

| Problem | Tip |
|--------|-----|
| Overfitting | Use augmentation, dropout |
| Input shape mismatch | Check resizing to expected input size |
| Low performance | Unfreeze top conv layers, adjust LR |
| Slow training | Use GPU (check CUDA) |
| Wrong loss | Use sparse vs categorical appropriately |

## ✅ Recommended Hyperparameters

| Component | Suggestion |
|-----------|------------|
| Learning rate | 1e-4 (head), 1e-5 (fine-tuned layers) |
| Batch size | 32 or 64 |
| Optimizer | Adam |
| Epochs | 10–30 |
| Image size | 224x224 |

## 📋 Final Checklist: "Have you checked everything?"

- [ ] CUDA (GPU) enabled?
- [ ] Random seeds set?
- [ ] Input shapes and preprocessing verified?
- [ ] Correct loss function?
- [ ] Fine-tuned layers properly unfrozen?
- [ ] Learning rates adjusted?
- [ ] Evaluation metrics in place?
- [ ] Save model checkpoints or use early stopping?
