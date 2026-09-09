# OIBSIP/DataScience-Task4-EmailSpamDetection

**Track:** Data Science | **Task:** 4 — Email Spam Detection with Machine Learning

## Objective
Build an NLP binary classifier distinguishing spam messages from legitimate (ham)
messages.

## Tech Stack
Python, pandas, scikit-learn (TF-IDF, Naive Bayes, Logistic Regression), Jupyter
Notebook

## ⚠️ Data note
Built without internet access, so the real "SMS Spam Collection" dataset (Kaggle /
UCI) could not be downloaded. The notebook generates a synthetic corpus of
spam-style and ham-style messages from templates, at a similar ~13% spam ratio to
the real dataset. **Before submitting, download the real dataset from Kaggle
("SMS Spam Collection Dataset") and replace the generated dataframe with
`pd.read_csv("spam.csv", encoding="latin-1")` (rename its columns to `label`,
`message`)** — no other code needs to change.

## Approach
- Class distribution check.
- Text preprocessing: lowercasing, punctuation removal.
- TF-IDF feature extraction (explained in the notebook).
- Trained Multinomial Naive Bayes and Logistic Regression.
- Evaluated with accuracy, precision, recall, F1, confusion matrix.
- Discussion of why recall matters most for spam detection.

## Files
- `Email_Spam_Detection.ipynb` — full executed notebook with outputs.
