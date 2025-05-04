# Sentiment-Prediction

This project builds a sentiment prediction system for text messages using machine learning and SentenceTransformer embeddings. It allows users to explore the full model development process and test the final model in an interactive Streamlit app.

---

## Project Overview

The workflow includes:

1. **Raw Data** – Starting from a dataset of text messages
2. **Filtering** – Keeping only English-language messages
3. **Labeling** – Using Gemini API to label ~30% of the data
4. **Embeddings** – Generating dense sentence embeddings using `all-mpnet-base-v2`
5. **Training** – Training a Logistic Regression model on the labeled data
6. **Semi-supervised Labeling** – Labeling the remaining data using the model
7. **Final Model** – Re-training on the full labeled dataset for best performance
8. **Deployment** – Interactive app built with Streamlit

---

## Technologies Used

- Python
- Streamlit
- scikit-learn
- SentenceTransformers
- Pandas
- Joblib
- Gemini (for initial labeling)
