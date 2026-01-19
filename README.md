# Brain Tumor Classification with CNNs (Keras)

This project implements convolutional neural networks (CNNs) for binary brain tumor classification using Keras with a TensorFlow backend.  
The goal is to compare different CNN architectures and evaluate their performance on an imbalanced medical image dataset.

---

## Dataset

The dataset is organized into training and test sets:
BrainTumorDataset/
|-- train/
| |-- no/
| |-- yes/
|---test/
|--- no/
|--- yes/

- Images are resized to **256×256**
- Binary classification: tumor / no tumor
- Class distribution is checked before training

---

## Models

Three CNN architectures are implemented and trained:

1. **Simple CNN** – baseline model  
2. **Deeper CNN** – more convolutional layers and parameters  
3. **BatchNorm CNN** – uses batch normalization for more stable training  

All models use:
- Adam optimizer  
- Binary cross-entropy loss  
- Metrics: accuracy, recall, precision  


## Training

- Each model is trained for 20 epochs
- Learning curves are plotted:
  - loss
  - accuracy
  - recall

Because the dataset is imbalanced, **recall** is treated as the main metric.

---

## Libraries

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
