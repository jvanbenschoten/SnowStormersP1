import pandas as pd
import ast, re
import numpy as np

df = pd.read_csv("LetterboxdTop250-5000reviews.csv")

def parse_list_cell(s):
    if pd.isna(s):
        return []
    return ast.literal_eval(s)

def rating_str_to_float(r):
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return np.nan
    r = str(r).strip()
    if r.lower() == "none" or r == "":
        return np.nan

    stars = r.count("★")
    half = 0.5 if "½" in r else 0.0

    if stars == 0 and half == 0.5:
        return 0.5
    if stars > 0:
        return stars + half

    return np.nan

rows = []
for _, row in df.iterrows():
    ratings = parse_list_cell(row["RATINGS"])
    reviews  = parse_list_cell(row["REVIEWS"])
    n = min(len(ratings), len(reviews))

    for j, (txt, rat) in enumerate(zip(reviews[:n], ratings[:n]), start=1):
        rating = rating_str_to_float(rat)
        if pd.isna(rating):
            continue

        text = str(txt).replace("\u00a0", " ").strip()
        text = re.sub(r"\s*\*{3}\s*$", "", text).strip()

        if len(text) == 0:
            continue

        rows.append({
            "movie_rank": int(row["Unnamed: 0"]),
            "movie_title": row["NAME"],
            "year": int(row["YEAR"]),
            "director": row["DIRECTOR"],
            "synopsis": row["SYNOPSYS"],
            "critic_id": j,          # placeholder per-movie index
            "star_rating": rating,   # numeric float
            "review_text": text,
            "review_char_len": len(text),
            "review_word_len": len(re.findall(r"\b\w+\b", text)),
        })

clean = pd.DataFrame(rows)
clean.to_csv("letterboxd_top250_reviews_clean.csv", index=False)
print(clean.shape)
print(clean.head())

pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
