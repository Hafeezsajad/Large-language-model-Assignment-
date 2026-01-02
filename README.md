# Large-language-model-Assignment-
Assignment 3  Large language model
### Sentiment Analysis on Amazon Reviews using RoBERTa
## Introduction
This project is based on sentiment analysis, which means identifying whether a text is positive or negative.

In this project, Amazon product reviews are used to train and test sentiment classification models.
Two approaches are compared:

A fine-tuned RoBERTa model

A baseline machine learning model using TF-IDF and Logistic Regression

The purpose of this project is to show that transformer-based models perform better than traditional methods.

### Dataset

Dataset Name: Amazon Polarity Reviews

### Source: Hugging Face

Classes:

0 → Negative review

1 → Positive review

## Data Used

Training data: 100,000 reviews

Testing data: 20,000 reviews

The dataset is balanced, meaning both classes have similar numbers of samples.

### Methodology

## Data Preprocessing

Review text is tokenized using RoBERTa tokenizer

Padding and truncation are applied

Maximum sequence length is set to 128

Data is converted into PyTorch format

## RoBERTa Model

Model used: RoBERTaForSequenceClassification

Number of output classes: 2

The model is fine-tuned on the training dataset

Training Settings

Learning rate: 2e-5

Batch size: 16

Epochs: 3

Optimizer: AdamW

Loss function: Cross-Entropy Loss

## Baseline Model

A simple baseline model is implemented for comparison:

TF-IDF Vectorizer (max features = 5,000)

Logistic Regression (max iterations = 1,000)

The same dataset split and evaluation metrics are used for fair comparison.

### Evaluation Metrics

The following metrics are used:

Accuracy

Precision

Recall

F1-Score

### Results
Model Performance
Model	Accuracy	Precision	Recall	F1-Score
TF-IDF + Logistic Regression	0.8670	0.8650	0.8763	0.8706
Fine-Tuned RoBERTa	0.9515	0.9514	0.9515	0.9514

## The RoBERTa model performs better than the baseline model in all metrics.

### Discussion

RoBERTa understands context and meaning of words

TF-IDF only counts words and ignores context

Transformer models give higher accuracy in sentiment analysis tasks

### Limitations

Only positive and negative labels are used

Neutral sentiment is not included

RoBERTa requires more computation and GPU support

Model is trained on a subset of the dataset

### Future Work

Use full dataset

Try other models like BERT or DistilBERT

Perform hyperparameter tuning

Use multi-class sentiment analysis

▶️ How to Run the Project
Step 1: Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Step 2: Install Required Libraries
pip install -r requirements.txt


Or manually:

pip install torch transformers datasets scikit-learn numpy pandas matplotlib

Step 3: Run Baseline Model
python baseline.py


This script trains and evaluates the TF-IDF + Logistic Regression model.

Step 4: Run RoBERTa Model
python train_roberta.py


This script fine-tunes the RoBERTa model and evaluates its performance.

## GPU is recommended for faster training.

📁 Project Structure
├── baseline.py
├── train_roberta.py
├── evaluate.py
├── requirements.txt
├── README.md

### Tools and Libraries Used

Python

PyTorch

Hugging Face Transformers

Scikit-learn

NumPy

Pandas

### Author

Student Assignment Project
Sentiment Analysis using NLP
