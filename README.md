# Snow Stormers — Interpreting Opinions: How Reviews Translate to Ratings

This repository contains the data pipeline, analysis code, and documentation for our DS 4002 project evaluating how accurately a pretrained Hugging Face sentiment model predicts sentiment in Letterboxd reviews (and how performance varies by review text characteristics).

This project uses a cleaned, review-level dataset derived from a Kaggle Letterboxd Top 250 dataset (4,755 reviews across 250 films)

## Contents of this Repository
At a high level, this repo supports a full workflow:
1) start from a cleaned, review-level Letterboxd dataset,  
2) convert star ratings into sentiment categories (negative/neutral/positive),  
3) run a pretrained Hugging Face transformer model on review text,  
4) apply a neutral confidence threshold,  
5) compute evaluation metrics (accuracy, macro F1, confusion matrix), and  
6) generate diagnostic plots (prediction distribution, confidence distribution, etc.).

---

## Section 1: Software and Platform

### Software Used
- **Python 3.x** (primary language for data handling, modeling, and evaluation)
- **Jupyter Notebook** or **VS Code** or **Google Colab**
  
### Platform Used
- Developed and tested in Google Colab (cloud-based Python environment).
- Compatible with Windows, Mac with correct package installation.

### Required Python Packages
Install the following packages (names shown for `pip`):
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- transformers
- tqdm
- emoji
- langdetect

---

## Section 2: Map of Documentation (Repository Structure)

      SnowStormersP1/
      │
      ├── data/
      ├── output/
      ├── scripts/
      ├── .gitignore
      ├── LICENSE.md
      ├── README.md
      └── how-to-write-a-readme.md

## Section 3: Instructions to Reproduce Results

Follow these steps to reproduce all model results and feature analysis outputs.

### Step 0 — Confirm You Have the Required Dataset
Make sure the dataset exists at:

data/LetterboxdTop250-5000reviews.csv

### Step 1 — Open Google Colab

Go to:

https://colab.research.google.com

Click New Notebook.

Google Colab runs Python in the cloud, so no local installation of Python is required.

### Step 2 — Upload Required Files

Upload the following into Colab (same working directory):

Scripts

clean_dataset.py

project_1_m13_analysis.py

Raw dataset

LetterboxdTop250-5000reviews.csv

### Step 3 — Install Required Packages in Colab

!pip install pandas numpy matplotlib seaborn scikit-learn transformers tqdm emoji langdetect

### Step 4 — Run the Data Cleaning Script

In a new code cell, run:

!python clean_dataset.py

This script will:

- Load the raw Kaggle file: LetterboxdTop250-5000reviews.csv

- Parse the embedded list columns for ratings and reviews

- Convert star strings (★ and ½) into numeric ratings (0.5–5.0)

- Remove missing ratings and empty reviews

- Export the cleaned review-level dataset:

### Step 5 — Run the Model + Feature Engineering Script

In a new code cell, run:

!python project_1_m13_analysis.py

This script will:

- Model pipeline

- Load letterboxd_top250_reviews_clean.csv

- Convert star ratings into sentiment labels:

        < 2.5 → negative

        2.5–3.5 → neutral

        3.5 → positive

- Run Hugging Face model:

      cardiffnlp/twitter-roberta-base-sentiment-latest

- Apply neutral confidence threshold (0.4–0.6)

- Compute:

      Accuracy

      Macro F1 score

      Confusion matrix

      Classification report

      Feature engineering

      Character length, word count

      Emoji count + emoji presence

      Caps ratio

      URL presence

      Exclamation/question counts

      Language detection

      Alphabetic ratio + low text content flag

- Exports

      full_feature_engineering_table.csv

      model_performance.csv

      confusion_matrix.csv

      confusion_matrix.png

      accuracy_by_language.csv

      accuracy_by_emoji.csv

      accuracy_by_length.csv

      accuracy_by_caps_ratio.csv

      accuracy_by_low_text_content.csv

- Visualizations

      Confusion matrix heatmap

      Predicted sentiment distribution

      Star ratings by predicted sentiment

      Confidence score distribution

      Confidence vs correctness

      True sentiment distribution

      Accuracy-by-feature plots (length, caps, emoji, language, low text, confidence vs length)
