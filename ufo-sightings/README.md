# UFO Sightings Analysis 🛸🌌

## Overview
This project explores a dataset of UFO sightings reported across the United States. It includes two main tasks:
1. **Predictive Modeling**: Forecast the number of sightings in a given year/state.
2. **Textual Analysis (NLP)**: Analyze sighting descriptions to uncover linguistic patterns and themes.

Dataset Source: [NUFORC UFO Sightings – Kaggle](https://www.kaggle.com/datasets/NUFORC/ufo-sightings)

---

## Project Structure
- `task1_predictive_model.ipynb` – Jupyter notebook for Task 1
- `task2_text_analysis.ipynb` – Jupyter notebook for Task 2
- `data/` – Cleaned subset of the dataset, post preprocessing (date parsing, filtering invalid records, generating features)
- `outputs/` – Graphs, tables, and visualizations
- `README.md` – Project summary

---

## Setup & Dependencies
This project uses Python 3. Dependencies include:
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn
- nltk / spacy
- sentence-transformers
- wordcloud

To install dependencies, run:
```bash
pip install -r requirements.txt
```
> (Ensure that your `requirements.txt` lists all the packages above.)

---

## Task 1: Predictive Modeling of Sightings
**Goal**: Predict the number of sightings in a given state/year.

### Steps Taken:
1. **Data Cleaning**:
   - Parsed date columns and extracted year
   - Removed rows with missing or malformed values
   - Applied log-transform to sighting duration

2. **Feature Engineering**:
   - Created proportions of each UFO shape out of total sightings per state/year
   - Time features: day/night indicator, weekday/weekend, season
   - Rolling average of sightings per state (lag of 1–3 years) to capture temporal trends

3. **Modeling**:
   - Trained Random Forest Regressor with train-test split (75/25)
   - Evaluated with RMSE, MAE, and R² metrics

### Results:
- RMSE: 31.17  
- MAE: 9.86  
- R²: 0.695

### Feature Importance:
Top features impacting the prediction:
- `cross_prev3`, `cone_prev3`, `chevron_prev3`: relative frequency of specific UFO shapes ("cross", "cone", "chevron") from 3 years prior. These indicate repeating patterns.
- `flash_prev3`, `teardrop_prev3`, `changing_prev3`: additional shape-related frequencies.
- `avg_log_duration_prev3`: log average of sighting duration 3 years back.
- `night_prev3`, `weekday_prev3`: indicators capturing temporal behavior.

### Model Evaluation:
- Model performance was strong, with no significant overfitting (train R² = 0.742, test R² = 0.695).
- Generalized well across states/years due to inclusion of lag features and conservative tree depth.

---

## Task 2: Textual Analysis of Sighting Descriptions
**Goal**: Identify recurring words, patterns, and clusters in the textual comments.

### Methods:
1. **Text Preprocessing**:
   - Lowercasing, punctuation removal, tokenization
   - Stopword removal, lemmatization

2. **Vectorization & Analysis**:
   - CountVectorizer and TF-IDF for word frequency
   - Word clouds for daytime vs nighttime comparisons
   - Shape-color heatmap

3. **Clustering with Sentence-BERT**:
   - Sentence embeddings for semantic similarity
   - KMeans clustering (k=5)

### Observations:
- Nighttime descriptions used more visual and color terms.
- No strong clustering patterns emerged by state.
- Embeddings showed common vocabulary across all clusters.

---

## Conclusions
- **Task 1**: The model is accurate and highlights historical shape frequencies as strong predictors. Performance could be further improved with richer context features.
- **Task 2**: NLP methods showed style and theme differences (e.g., day/night), but clustering was inconclusive due to noisy or uniform text patterns.

## Recommendations for Future Work
- Consider using cross-validation techniques or time-based train/test splits to better evaluate model generalization over years.
- Use XGBoost and compare with Random Forest.
- Consider external features (e.g., population per state).
- Improve preprocessing by filtering overly short/long reports.
- Expand temporal scope – e.g., combine similar shapes into categories.
- Explore topic modeling (LDA, BERTopic) for deeper text pattern extraction.

---

## Author
**Noam Yehoshua**  
B.Sc. Student in Industrial Engineering and Data Science  
Ben-Gurion University of the Negev