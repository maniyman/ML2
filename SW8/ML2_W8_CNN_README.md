
# 📖 README – Machine Learning 2, SW 8: Convolutional Neural Networks

## Overview

This README summarizes the key content from **Week 8 (SW 8)** of the Machine Learning 2 course (FS 2025), based on the lecture slides provided by Dr. Elena Gavagnin. The focus is on **classic deep learning methods for image data**, especially **Convolutional Neural Networks (CNNs)**.

---

## Topics Covered

✅ Artificial Neural Networks (ANN) introduction and TensorFlow/Keras implementation  
✅ Convolutional Neural Networks (CNN) basics and architecture  
✅ Famous CNN architectures (e.g., LeNet, AlexNet, ImageNet)  
✅ Transfer Learning and Fine Tuning with pre-trained layers  
✅ Visual data preparation (image shapes, channels, preprocessing)  
✅ Optimizers (SGD, Adam, others)  
✅ Regularization techniques: L2, L1, Dropout  
✅ Early Stopping to avoid overfitting  
✅ TensorBoard for live visualization and tracking

---

## Key Resources

- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html  
  (also directly loadable via `tf.keras.datasets.cifar10`)
- TensorBoard: https://www.tensorflow.org/tensorboard
- Stanford CS231n CNN Visual Recognition: http://cs231n.stanford.edu/
- Géron, A. – Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (Ch. 11, 14)

---

## Main Hands-on Task (Exercise)

- Adapt the MNIST example (28x28 grayscale) to CIFAR-10 (32x32 RGB).
- Modify the input layer and architecture to handle 3 channels (RGB).
- Train and evaluate performance.
- Compare results to MNIST.
- Use TensorBoard to monitor training.
- Apply regularization techniques.
- Investigate optimizers beyond SGD.

---

## Notes

- **Input Shapes**: MNIST (28,28,1), CIFAR-10 (32,32,3)  
- **Output Classes**: 10 for both datasets
- **Common Activation Functions**: ReLU (hidden layers), Softmax (output layer)
- **Loss Function**: Sparse Categorical Crossentropy (for integer labels)

---

## Contacts

Instructor: Dr. Elena Gavagnin  
Email: elena.gavagnin@zhaw.ch

---

For detailed practice, please refer to the accompanying cheat sheet and the provided notebooks.

