
# 📚 Machine Learning 2 – Week 8 Cheat Sheet: Convolutional Neural Networks (CNNs)

## ✅ Key Concepts

- **Artificial Neural Network (ANN)**: A layered model of interconnected neurons for learning from data.
- **Convolutional Neural Network (CNN)**: Specialized neural network for image data; uses convolutional and pooling layers to extract spatial features.
- **Convolution**: Operation applying a learnable filter (kernel) over an input to create feature maps.
- **Pooling (Subsampling)**: Reduces the spatial dimensions (width, height) of feature maps, commonly using max or average.
- **Receptive Field**: The region of the input image a particular neuron is sensitive to.
- **Transfer Learning**: Using a pre-trained model (e.g., on ImageNet) and fine-tuning it for a new task.
- **TensorBoard**: Visualization tool for tracking metrics, losses, and model graphs.
- **Regularization**: Techniques like L2 regularization, dropout, and early stopping to prevent overfitting.

## ✅ Typical Task Patterns

### 🔹 Basic CNN Pipeline
1. Load dataset (e.g., MNIST, CIFAR10).
2. Preprocess images (reshape, normalize).
3. Define CNN architecture.
4. Compile model with loss, optimizer, and metrics.
5. Train with `.fit()` and optionally validate.
6. Evaluate with `.evaluate()`.
7. Predict/infer with `.predict()`.

### 🔹 Common Tasks in Exercises
- Adapting models from grayscale (MNIST, 28x28x1) to RGB datasets (CIFAR10, 32x32x3).
- Adjusting input shapes.
- Monitoring training using TensorBoard.
- Applying dropout and L2 regularization.
- Implementing early stopping.
- Comparing performance on different datasets.

## ✅ Example Code Snippets

### 🛠 Load and Preprocess CIFAR10
```python
import tensorflow as tf

# Load CIFAR10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Check shapes
print(x_train.shape)  # (50000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)
```

### 🛠 Define a CNN Model
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes
])
```

### 🛠 Compile and Train
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))
```

### 🛠 Apply Regularization
```python
from tensorflow.keras import regularizers

# Add L2 regularization
tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                       kernel_regularizer=regularizers.l2(0.01))
```

### 🛠 Add Dropout
```python
tf.keras.layers.Dropout(0.5)
```

### 🛠 Early Stopping
```python
early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

model.fit(x_train, y_train, epochs=30,
          validation_data=(x_test, y_test),
          callbacks=[early_stop])
```

### 🛠 TensorBoard Logging
```python
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs')

model.fit(x_train, y_train, epochs=10,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_cb])
```

## ✅ Relevant Libraries and Frameworks

- **TensorFlow/Keras**: Main framework for building and training models.
- **NumPy**: For data manipulation.
- **Matplotlib/Seaborn**: For plotting results.
- **TensorBoard**: For visualization.
- **Scikit-learn** (optional): For evaluation metrics beyond accuracy.

## ✅ Best Practices & Common Pitfalls

| ⚡ Best Practices                         | ⚠ Common Pitfalls                          |
|------------------------------------------|-------------------------------------------|
| Normalize input data (scale to [0, 1])   | Forgetting to adjust input shape for CNNs  |
| Set random seeds for reproducibility     | Ignoring overfitting (watch val loss)      |
| Use callbacks like EarlyStopping         | Using too large/complex models on small data|
| Monitor with TensorBoard                 | Forgetting to shuffle or split datasets    |
| Check GPU availability (for speed)       | Misaligning label formats (sparse vs. one-hot)|

## ✅ Recommended Hyperparameters (Starting Points)

- **Optimizer**: Adam (default)
- **Learning Rate**: 0.001
- **Batch Size**: 32 or 64
- **Dropout Rate**: 0.2–0.5
- **L2 Regularization**: 0.001–0.01
- **Epochs**: Start with 10–30, adjust based on early stopping.

## ✅ Final Checklist: Have You Checked Everything?

✅ Input data shapes match model expectations?  
✅ Labels are in the correct format (e.g., sparse integers)?  
✅ Random seeds set for reproducibility?  
✅ CUDA/GPU enabled if available?  
✅ Batch size fits into GPU memory?  
✅ Early stopping or checkpointing in place?  
✅ Training/validation loss monitored (no overfitting)?  
✅ Model saved after training (`model.save()`)?  
✅ Code tested on a small batch before full run?
