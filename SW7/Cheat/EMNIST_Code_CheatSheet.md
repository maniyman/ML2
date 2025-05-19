# 📘 EMNIST Code Cheat Sheet (Kernfunktionen aus den Notebooks)

## ⚙️ Modellaufbau, Training, Evaluation & Speicherung

### 🧩 Code-Snippet 1
```python
#setup folder where you will save logs for tensorflow
root_logdir = os.path.join(os.curdir,"my_logs_ML2")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
```

### 🧩 Code-Snippet 2
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # EMNIST images are 28x28 with 1 channel
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(26, activation = 'softmax')  # 26 classes for letters A-Z
])
```

### 🧩 Code-Snippet 3
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
```

### 🧩 Code-Snippet 4
```python
history = model.fit(train_images, train_labels, epochs=30,
                    validation_split=0.1,
                    callbacks=[tensorboard_cb])
```

### 🧩 Code-Snippet 5
```python
model.save("best_model_emnist.keras")
```

### 🧩 Code-Snippet 6
```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```

### 🧩 Code-Snippet 7
```python
# Select a few test images
X_new = test_images[:3]

# Make predictions
y_proba = model.predict(X_new)
y_pred = np.argmax(y_proba, axis=1)

print("Predicted classes:", y_pred)
print("Predicted letters:", [letters[pred] for pred in y_pred])
print("Actual classes:", test_labels[:3])
print("Actual letters:", [letters[label] for label in test_labels[:3]])
```

### 🧩 Code-Snippet 8
```python
# Select the first 15 test images
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*num_cols, 2*num_rows))

# Get predictions for all test images
predictions = model.predict(test_images)

for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(letters[predicted_label],
                              letters[true_label]),
                              color=color)
plt.tight_layout()
plt.show()
```

### 🧩 Code-Snippet 9
```python
plt.figure(figsize=(10,3))
# Plot training & validation accuracy values
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.show()
```

### 🧩 Code-Snippet 10
```python
def build_model(hp):
    model = tf.keras.Sequential([
        # Flatten the input images
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),

        # First dense layer with tunable units (similar to original MNIST example)
        tf.keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=128, max_value=512, step=64),
            activation='relu',
            kernel_initializer=tf.keras.initializers.he_normal
        ),

        # Second dense layer with tunable units
        tf.keras.layers.Dense(
            units=hp.Int('dense_2_units', min_value=64, max_value=256, step=32),
            activation='relu',
            kernel_initializer=tf.keras.initializers.he_normal
        ),

        # Optional third dense layer
        tf.keras.layers.Dense(
            units=hp.Int('dense_3_units', min_value=32, max_value=128, step=32),
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(
                hp.Float('l2_reg', min_value=0.001, max_value=0.1, sampling='log')
            )
        ),

        # Dropout layer with tunable rate
        tf.keras.layers.Dropout(
            rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        ),

        # Output layer - 26 classes (A-Z) with softmax activation
        tf.keras.layers.Dense(26, activation='softmax')
    ])

    # Compile with tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-3, max_value=1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

### 🧩 Code-Snippet 11
```python
# Initialize the tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='keras_tuner_dir',
    project_name='emnist_letters'
)

# Display search space summary
tuner.search_space_summary()
```

### 🧩 Code-Snippet 12
```python
# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Add model checkpoint callback
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_checkpoint.keras',
    save_best_only=True,
    monitor='val_accuracy'
)

# Perform the search
tuner.search(
    train_images, train_labels,
    validation_split=0.1,
    epochs=10,
    callbacks=[tensorboard_cb, early_stopping, checkpoint_cb]
)
```

### 🧩 Code-Snippet 13
```python
# Train the best model for more epochs
history = best_model.fit(
    train_images, train_labels,
    epochs=15,
    validation_split=0.1,
    callbacks=[tensorboard_cb, early_stopping, checkpoint_cb, lr_scheduler]
)
```

### 🧩 Code-Snippet 14
```python
# Evaluate on test set
test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

# Save the best model
best_model.save('best_emnist_letters_model.keras')
```

### 🧩 Code-Snippet 15
```python
# Create a writer for the custom TensorBoard data
custom_writer = tf.summary.create_file_writer(os.path.join(run_logdir, 'custom_metrics'))

# Log sample images with predictions
with custom_writer.as_default():
    # Get sample images
    sample_images = test_images[:10]
    sample_labels = test_labels[:10]

    # Make predictions
    predictions = best_model.predict(sample_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Convert images to right format for TensorBoard (add batch dimension)
    images_for_tb = np.reshape(sample_images, (-1, 28, 28, 1))

    # Add titles with true and predicted labels
    titles = [f"True: {letters[true]}, Pred: {letters[pred]}"
              for true, pred in zip(sample_labels, predicted_labels)]

    # Log images with predictions as titles
    tf.summary.image('Test Predictions', images_for_tb, max_outputs=10, step=0)
```

### 🧩 Code-Snippet 16
```python
# Plot training history
plt.figure(figsize=(10, 3))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.show()
```

### 🧩 Code-Snippet 17
```python
# Get some test samples
sample_indices = np.random.choice(len(test_images), 25, replace=False)
sample_images = test_images[sample_indices]
sample_labels = test_labels[sample_indices]

# Make predictions
predictions = best_model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Visualize predictions
plt.figure(figsize=(15, 15))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(sample_images[i].squeeze(), cmap='gray')

    # Get confidence for the prediction
    confidence = predictions[i][predicted_labels[i]] * 100

    color = 'green' if predicted_labels[i] == sample_labels[i] else 'red'
    plt.title(f'True: {letters[sample_labels[i]]}\nPred: {letters[predicted_labels[i]]}\nConf: {confidence:.2f}%',
              color=color, fontsize=9)
    plt.axis('off')
plt.tight_layout()
plt.show()
```

### 🧩 Code-Snippet 18
```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Make predictions on all test data
predictions = best_model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix - normalized version
plt.figure(figsize=(15, 12))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
            xticklabels=letters,
            yticklabels=letters)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(test_labels, predicted_labels,
                           target_names=letters))
```

