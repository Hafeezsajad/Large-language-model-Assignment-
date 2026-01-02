# Large-language-model-Assignment-
Assignment 3  Large language model
### Sentiment Analysis on Amazon Reviews using RoBERTa
## Project Overview

This project focuses on sentiment analysis of Amazon product reviews.
The goal is to automatically classify reviews as positive or negative.

Two different approaches are implemented and compared:

Fine-tuned RoBERTa model (Transformer-based deep learning)

TF-IDF + Logistic Regression (Baseline machine learning model)

This comparison helps demonstrate why transformer models perform better for text classification tasks.

### Dataset

The project uses the Amazon Polarity Reviews Dataset from Hugging Face.

Dataset Details:

Labels:

0 → Negative review

1 → Positive review

Training samples: 100,000

Test samples: 20,000

Dataset is approximately balanced between both classes

### Methodology
## Data Preprocessing

Text reviews are tokenized using the RoBERTa tokenizer

Padding and truncation are applied

Maximum sequence length is set to 128 tokens

Data is converted into PyTorch tensors

## RoBERTa Model

Model used: RoBERTaForSequenceClassification

Number of output classes: 2

Fine-tuned using the Hugging Face Transformers library

Training Hyperparameters:

Learning rate: 2e-5

Batch size: 16

Epochs: 3

Optimizer: AdamW

Loss function: Cross-Entropy Loss

## Baseline Model

A baseline classifier is implemented for comparison:

TF-IDF Vectorizer (max features = 5,000)

Logistic Regression (max iterations = 1,000)

The same dataset split and evaluation metrics are used for both models.

### Evaluation Metrics

The following metrics are used to evaluate performance:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

### Results
##Model Performance Comparison
Model	Accuracy	Precision	Recall	F1-Score
TF-IDF + Logistic Regression	0.8670	0.8650	0.8763	0.8706

Fine-Tuned RoBERTa	0.9515	0.9514	0.9515	0.9514

## Observation:
The RoBERTa model outperforms the baseline model across all evaluation metrics.

### Discussion

RoBERTa captures contextual meaning and word relationships

TF-IDF ignores word order and context

Transformer-based models provide higher accuracy for sentiment analysis

### Limitations

Only binary sentiment classification is used

Neutral sentiment is not included

RoBERTa requires GPU and higher computational resources

Model trained on a subset of the dataset


### Technologies Used

Programming Language: Python

Frameworks: PyTorch, Hugging Face Transformers

Libraries: NumPy, Pandas, Scikit-learn, Matplotlib

Platform: Google Colab / Local Machine

Version Control: GitHub

▶️ How to Run the Code
Clone the Repository
git clone <your-repo-link>
cd <your-repo-name>

Install Required Packages
pip install -r requirements.txt


Or install manually:

pip install torch transformers datasets scikit-learn numpy pandas matplotlib

Run Baseline Model
python baseline.py

Run RoBERTa Model
python train_roberta.py


## Note: GPU is recommended for faster training.

 Project Structure
├── baseline.py
├── train_roberta.py
├── evaluate.py
├── requirements.txt
├── README.md

## Applications

Customer review analysis

Product feedback monitoring

Opinion mining

E-commerce sentiment analysis

### Author

Student Assignment Project
Sentiment Analysis using NLP and Transformers
