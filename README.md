# Snow Stormers - Interpreting Opinions: How Reviews Translate to Ratings

This repository contains the code, data processing steps, and documentation for our DS 4002 project examining how features of Letterboxd review text influence the predictive accuracy of a pretrained Hugging Face sentiment analysis model.

## Contents of this Repository
This project uses a cleaned, review-level dataset derived from a Kaggle Letterboxd Top 250 dataset (4,755 reviews across 250 films). We (1) reshape and clean the raw movie-level data into review-level observations, (2) engineer review text features (emoji usage, capitalization, review length, language), (3) run a pretrained Hugging Face transformer sentiment model to generate sentiment labels, (4) map numeric star ratings to sentiment categories, and (5) evaluate model accuracy overall and across review subgroups.

---

## Section 1: Software and Platform

### Software Used
- **Python 3.x** (primary analysis and modeling)
- **Jupyter Notebook** or **VS Code** (recommended for running notebooks/scripts)
- *(Optional)* **Google Docs** (write-up/report drafting)

### Required Python Packages
Install the following packages (names shown as used in `pip`):
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `transformers`
- `torch`
- `langdetect` *(or alternative language detection package if used)*
- `regex` *(or Python built-in `re` may be sufficient for emoji detection)*

##Section 2:

Snow-Stormers-Letterboxd/
│
├── README.md
├── environment.yml / requirements.txt
│
├── data/
│   ├── raw/
│   │   └── letterboxd_top250_raw.csv                # original Kaggle dataset (movie-level)
│   ├── processed/
│   │   └── letterboxd_top250_reviews_clean.csv      # cleaned review-level dataset (4,755 reviews)
│
├── notebooks/
│   ├── 01_data_reshape_clean.ipynb                  # reshape list-columns → review-level rows
│   ├── 02_eda.ipynb                                 # EDA figures and summary stats
│   ├── 03_feature_engineering.ipynb                 # emoji, caps ratio, word/char length, language
│   ├── 04_huggingface_model.ipynb                   # run transformer sentiment predictions
│   ├── 05_evaluation.ipynb                          # accuracy/F1/confusion + subgroup evaluation
│
├── src/
│   ├── reshape_clean.py                             # script version of reshaping/cleaning
│   ├── features.py                                  # feature engineering helper functions
│   ├── model_inference.py                           # Hugging Face inference pipeline
│   ├── evaluation.py                                # metrics + subgroup comparison
│
├── results/
│   ├── figures/
│   │   ├── star_rating_distribution.png
│   │   ├── review_length_distribution.png
│   │   ├── review_length_vs_rating.png
│   │   ├── confusion_matrix.png
│   │   └── subgroup_accuracy.png
│   ├── tables/
│   │   ├── overall_metrics.csv
│   │   └── subgroup_metrics.csv
│
└── docs/
    ├── MI2_writeup.pdf / MI2_writeup.docx           # M2 document
    └── references.md                                # citations / reference links (optional)


##Section 3: Instructions to Reproduce Results
Follow the steps below to reproduce our results from raw data → cleaned dataset → model predictions → evaluation.
Obtain the Letterboxd Top 250 dataset from Kaggle (raw movie-level file).


Place the raw file in:


data/raw/letterboxd_top250_raw.csv


Confirm the repository also contains (or can generate) the processed dataset:


data/processed/letterboxd_top250_reviews_clean.csv


Step 1 — Set Up Python Environment
Install Python 3.x.


Create and activate a virtual environment (recommended).


Install dependencies:

 pip install -r requirements.txt
 (If requirements.txt is not available, install the packages listed in Section 1.)


Step 2 — Reshape and Clean Data (Movie-level → Review-level)
Goal: Convert list-formatted reviews/ratings into individual rows and create review length variables.
Open and run:


notebooks/01_data_reshape_clean.ipynb
 (or run python src/reshape_clean.py if using scripts)


This step should:


Expand each movie row into multiple review rows


Ensure each review aligns with its star rating


Remove empty/missing review text


Create:


review_char_len


review_word_len


Output should be saved to:


data/processed/letterboxd_top250_reviews_clean.csv


Step 3 — Run EDA Visualizations
Goal: Reproduce the three main EDA figures used in MI2.
Open and run:


notebooks/02_eda.ipynb


Generate and save figures:


Distribution of star ratings by number of reviews


Distribution of review length by number of reviews


Scatterplot: review length vs. star rating


Save figures to:


results/figures/


Step 4 — Feature Engineering (Text Characteristics)
Goal: Create review text features used to evaluate performance differences.
Open and run:


notebooks/03_feature_engineering.ipynb


Create variables including:


Emoji usage (binary indicator and/or count)


Capitalization ratio (proportion of uppercase characters/tokens)


Language detection label (English vs non-English, or full language code)


Save the feature-enhanced dataset (either overwrite processed file or create a new one), e.g.:


data/processed/letterboxd_top250_reviews_features.csv


Step 5 — Generate Sentiment Predictions (Hugging Face Model)
Goal: Use a pretrained transformer model to predict sentiment labels per review.
Open and run:


notebooks/04_huggingface_model.ipynb


Model used:


cardiffnlp/twitter-roberta-base-sentiment-latest


For each review, generate:


Predicted label: negative / neutral / positive


Confidence scores or probabilities (if captured)


Save predictions to:


results/tables/predictions.csv
 (or append predictions to the processed dataset)


Step 6 — Convert Star Ratings to Sentiment Categories
Goal: Make numeric ratings comparable to model outputs.
Use preregistered cutoffs to convert star_rating → true_label (neg/neu/pos).


Ensure these cutoffs are explicitly coded in:


notebooks/05_evaluation.ipynb
 (or src/evaluation.py)


Step 7 — Evaluate Model Performance (Overall + Subgroups)
Goal: Measure predictive accuracy and how it changes across review characteristics.
Open and run:


notebooks/05_evaluation.ipynb


Compute:


Overall accuracy


F1 score (macro or weighted, depending on class imbalance handling)


Confusion matrix


Subgroup comparisons:


With vs without emojis


Short vs long reviews (by threshold such as median word count)


High vs low capitalization (by threshold such as median caps ratio)


English vs non-English reviews


Save outputs:


results/tables/overall_metrics.csv


results/tables/subgroup_metrics.csv


results/figures/confusion_matrix.png


results/figures/subgroup_accuracy.png


Expected Reproduction Outcome
If steps are run successfully, you should reproduce:
The MI2 EDA figures (ratings distribution, review length distribution, review length vs rating)


Sentiment predictions for each review from the pretrained Hugging Face model


Overall and subgroup performance metrics (accuracy, F1, confusion matrix)


A summary identifying at least one review text characteristic associated with meaningful performance differences
