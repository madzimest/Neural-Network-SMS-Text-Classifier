# SMS Spam Classifier using Neural Networks

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![freeCodeCamp](https://img.shields.io/badge/freeCodeCamp-project-brightgreen)](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/neural-network-sms-text-classifier)

A binary text classifier that distinguishes between **ham** (legitimate) and **spam** messages using a Bidirectional LSTM neural network. This project was completed as part of the freeCodeCamp Machine Learning with Python curriculum.

## Project Goal

Build a machine learning model that takes an SMS message as input and returns:
- A probability score between 0 and 1 (0 = ham, 1 = spam)
- A label: either `"ham"` or `"spam"`

The model is evaluated on a held-out validation set and must pass freeCodeCamp's test suite.

## Dataset

The [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) is used, provided as two tab-separated files:

- `train-data.tsv` – 4,825 labeled messages for training
- `valid-data.tsv` – 1,207 labeled messages for validation

Each file contains two columns:
- `label` `ham` or `spam`
- `text` the raw SMS content

**Class distribution (training set):**
- ham: 4,255 (88.2%)
- spam: 570 (11.8%)

The dataset is slightly imbalanced, but the model handles it well with binary cross‑entropy loss.

## Methodology

### 1. Preprocessing
- **Label encoding**: `ham` → 0, `spam` → 1 using a single `LabelEncoder` fitted on training data only.
- **Tokenization**: `Tokenizer` from Keras with an `<OOV>` token to handle unseen words during inference.
- **Sequence length**: The 95th percentile of message lengths in the training set is used as `maxlen` (approx. 120 tokens). All sequences are padded/truncated to this length.
- **Padding**: Post-padding and post-truncating are applied for consistency.

### 2. Model Architecture
A **Bidirectional LSTM** network captures context from both directions of the text:

```python
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(1, activation='sigmoid')
])
