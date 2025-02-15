# Fake News Classifier Using LSTM

## Description
This project implements a **Fake News Classifier** using a **Long Short-Term Memory (LSTM)** model. The classifier aims to detect fake news articles based on textual content, leveraging deep learning-based NLP techniques for classification.

## Dataset
The dataset used for training and testing can be found on Kaggle:
[Fake News Dataset](https://www.kaggle.com/c/fake-news/data#)

## Overview
This Jupyter Notebook covers:
- **One-Hot Representation**: Encoding words as unique integers.
- **Embedding Representation**: Using word embeddings to improve feature representation.
- **Model Training**: Training an LSTM-based deep learning model for classification.
- **Performance Metrics & Accuracy**: Evaluating model performance using accuracy, precision, recall, and F1-score.

## Requirements
To run this notebook, install the required dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn nltk
```

## Usage
1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook FakeNewsClassifierUsingLSTM.ipynb
   ```
2. Execute each cell step by step to train and evaluate the model.
3. Modify the dataset or model parameters to experiment with different configurations.

## Implementation Details
- **TensorFlow/Keras**: Used for building and training the LSTM model.
- **Pandas & NumPy**: Used for data processing.
- **NLTK**: Used for text preprocessing.
- **Scikit-learn**: Used for performance evaluation.


