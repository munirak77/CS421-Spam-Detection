# CS421-Spam-Detection

This project compares machine learning and deep learning models for spam email detection.

Course: CS421  
Track: Track 2 - Comparative Study  
Group: 1

## Project Goal

The goal is to compare three models using the same email dataset, data split, and evaluation metrics.

Each email is classified as:
- Ham (0): a normal email.
- Spam (1): an unwanted email.

## Models

1. Logistic Regression with TF-IDF: the baseline.
2. Linear SVM with TF-IDF.
3. Text CNN with word embeddings.

## Dataset

The project uses the [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/).

Download these five files and keep their original names:

- [20030228_easy_ham.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2)
- [20030228_easy_ham_2.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2)
- [20030228_hard_ham.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2)
- [20030228_spam.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2)
- [20050311_spam_2.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2)

The notebook prepares the email subject and body, removes HTML tags, handles reading errors, and removes exact duplicate text.

After preparation, the dataset contains 5,915 emails:
- Ham: 4,108.
- Spam: 1,807.

The data is split into:
- Training/development: 4,732 emails.
- Final test: 1,183 emails.

All models use the same five cross-validation folds. The data split and fold assignments use random seed 42.

## How to Run

1. Open `Spam_Detection_Phase2 (3).ipynb` in Google Colab.
2. Install the needed packages in a setup cell:

   ```python
   %pip install pandas==2.2.3 numpy==2.4.6 scikit-learn==1.6.1 tensorflow==2.21.0 beautifulsoup4 matplotlib joblib
   ```

3. Restart the runtime if Colab asks you to.
4. Upload the five dataset archives to `/content/`. Keep them compressed.
5. Run the notebook cells in order, from top to bottom.
6. Download the backup files created by the notebook.

The recorded run used Python 3.13.18 and ran on CPU without a GPU.

Colab runtime files are temporary, so keep the downloaded backups.

## Final Test Results

The final test contains 1,183 emails.

All scores below use the 0-1 scale. Precision, Recall, and F1 are for Spam. Lower FPR is better.

| Model | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9738 | 0.9769 | 0.9363 | 0.9562 | 0.0097 |
| Linear SVM | 0.9899 | 0.9781 | 0.9889 | 0.9835 | 0.0097 |
| Text CNN | 0.9763 | 0.9612 | 0.9612 | 0.9612 | 0.0170 |

Linear SVM had the highest Accuracy and F1 in this experiment.

## Project Files

- `Spam_Detection_Phase2 (3).ipynb`: the experiment notebook.
- `models/`: the saved models and CNN input settings.
- `results/`: scores, predictions, confusion matrices, fold assignments, and error analysis.

The full experiment settings are saved in `results/experiment_settings.json`.

Keep `cnn_vocabulary.json` and `cnn_preprocessing.json` with the saved CNN model. They are needed to prepare its input correctly.

## Study Limits

The project uses one historical dataset and removes only exact duplicates.

CNN uses up to 300 tokens from each email, while TF-IDF uses the full prepared text. This difference may affect the comparison.

Results and training times may change with another dataset or computing environment. Final test examples were used for discussion, not for tuning the models.
