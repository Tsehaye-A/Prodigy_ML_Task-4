# Task 4: Handwritten Character Recognition with a Fully Connected Neural Network

This project demonstrates a **fully connected (dense) neural network** built using **Keras** to classify **handwritten characters** (A-Z, 26 classes) from a custom dataset. The model is trained on a 28×28 grayscale image dataset similar to MNIST but containing uppercase English letters.

---

## Dataset

- **Training Set**: `train.csv` – 27,455 samples
- **Test Set**: `test.csv` – 7,172 samples
- **Image Size**: 28×28 pixels (grayscale)
- **Classes**: 26 (A to Z → labeled 0 to 25)
- **Format**: Each row contains:
  - `label`: integer from 0 to 25
  - `pixel1` to `pixel784`: flattened pixel values (0–255)

> The dataset is downloaded from Google Drive using PyDrive.

---

## Model Architecture

A **fully connected neural network** with the following layers:

```python
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))
