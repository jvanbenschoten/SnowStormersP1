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
- Developed and tested on **Windows** (also compatible with Mac/Linux if Python + packages install correctly).

### Required Python Packages
Install the following packages (names shown for `pip`):
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `transformers`
- `tqdm`

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

Follow these steps to reproduce the MI3 results (metrics + plots) from the cleaned dataset.

### Step 0 — Confirm You Have the Required Dataset
Make sure the cleaned dataset exists at:

data/letterboxd_top250_reviews_clean.csv

### Step 1 — Open Google Colab

Go to:
https://colab.research.google.com

Click “New Notebook.”

Google Colab runs Python in the cloud, so no local installation of Python is required.

### Step 2 — Upload Required Files

In the Colab notebook:

On the left sidebar, click the folder icon.

Click Upload.

Upload:

mi3_analysis.py (from the scripts/ folder)

letterboxd_top250_reviews_clean.csv (from the data/ folder)

Both files must be in the same working directory inside Colab.

### Step 3 — Install Required Packages in Colab

!pip install pandas numpy matplotlib seaborn scikit-learn transformers tqdm

### Step 4 — Run the Analysis Script

In a new code cell, run:

!python mi3_analysis.py

This script will:

Load the cleaned dataset

Remove empty review text

Convert star ratings to sentiment labels:

### < 2.5 → negative

### 2.5–3.5 → neutral

### > 3.5 → positive

Run the pretrained Hugging Face model:
cardiffnlp/twitter-roberta-base-sentiment-latest

Apply a neutral confidence threshold (0.4–0.6)

Compute:

Accuracy

Macro F1 score

Confusion matrix

Classification report

Display six diagnostic plots:

Confusion matrix heatmap

Distribution of predicted sentiment

Boxplot: star ratings by predicted sentiment

Histogram: confidence scores

Boxplot: confidence (correct vs incorrect)

Distribution of true sentiment labels
