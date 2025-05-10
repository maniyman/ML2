# 🧠 ML2 Week 7 Cheat Sheet: Artificial Neural Networks (ANNs) & Practice Patterns

## ✅ Key Concepts

- **Artificial Neural Network (ANN)**: Model inspired by biological neurons; consists of layers of nodes (neurons) connected by weights, with activation functions introducing nonlinearity.
- **Perceptron**: Basic neural unit performing weighted sum + activation; works only on linearly separable data.
- **Multilayer Perceptron (MLP)**: Feedforward neural network with multiple hidden layers; allows learning of non-linear patterns.
- **Activation Functions**:
  - Linear: `f(x) = x` → not useful for deep learning.
  - Sigmoid: `σ(x) = 1 / (1 + e^{-x})` → squashes output between 0–1.
  - ReLU: `max(0, x)` → introduces sparsity, accelerates training.
  - Softmax: Converts logits to probabilities over multiple classes.
- **Loss Functions**:
  - BinaryCrossentropy → binary classification.
  - CategoricalCrossentropy → multi-class classification.
  - MeanSquaredError → regression.
- **Optimizers**: Algorithms to update weights: SGD, Momentum, RMSProp, Adam.
- **Regularization**: Techniques (L1/L2 penalties, dropout) to prevent overfitting.
- **Hyperparameters**: Settings like learning rate, batch size, epochs, layer sizes (not learned from data).

---

## ✅ Typical Task Patterns

### 🔹 1. Build and Compile a Model
```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=784),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 🔹 2. Training Loop
```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val)
)
```

### 🔹 3. Evaluation and Inference
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
predictions = model.predict(X_new)
```

### 🔹 4. Hyperparameter Tuning
```python
from kerastuner import RandomSearch

tuner = RandomSearch(
    build_model_fn,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir'
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

### 🔹 5. Regularization Techniques
```python
tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))
tf.keras.layers.Dropout(0.5)
```

---

## ✅ Relevant Libraries & Frameworks

| Purpose                     | Library             |
|-----------------------------|---------------------|
| Core NN framework           | TensorFlow, Keras   |
| Alternative framework       | PyTorch             |
| Hyperparameter tuning       | Keras Tuner, Hyperopt |
| Data handling               | NumPy, Pandas       |
| Image processing (if needed)| OpenCV, PIL         |
| Visualization               | Matplotlib, TensorBoard |

---

## ✅ Best Practices & Common Pitfalls

- ✅ Set random seeds for reproducibility (`tf.random.set_seed(42)`).
- ✅ Normalize/scale input data (e.g., divide pixel values by 255).
- ✅ Check input/output shapes (especially with `Flatten()` or convolutions).
- ✅ Use early stopping or model checkpoints to prevent overfitting.
- ✅ Monitor both training and validation metrics.
- ✅ Avoid too high learning rates → divergence.
- ✅ Avoid too small learning rates → very slow convergence.
- ✅ Check class balance → accuracy might be misleading if imbalanced.

---

## ✅ Recommended Hyperparameters

| Hyperparameter       | Typical Range/Value          |
|----------------------|-----------------------------|
| Learning rate        | 0.001–0.01 (Adam), 0.1–1 (SGD) |
| Batch size          | 32, 64, 128                 |
| Epochs              | 10–50 (watch for overfit)   |
| Dropout rate        | 0.2–0.5                    |
| Optimizer           | Adam (default), SGD + momentum for experimentation |

---

## ✅ Final Checklist: Have You Checked Everything?

✅ Is CUDA enabled if using GPU?  
✅ Are random seeds set for reproducibility?  
✅ Are data shapes (input/output) correctly matched?  
✅ Did you split data into train, validation, and test?  
✅ Are loss functions and metrics appropriate for your task (classification vs regression)?  
✅ Did you monitor learning curves (loss/accuracy) over epochs?  
✅ Did you save model checkpoints or the final model?  
✅ Did you test your model on unseen data?

---
