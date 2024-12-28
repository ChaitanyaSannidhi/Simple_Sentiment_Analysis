# Sentiment Analysis Through Twitter Tweets

This project focuses on performing **sentiment analysis** on Twitter tweets using machine learning techniques. The model classifies tweets into categories such as positive, negative, or neutral sentiment.

---

## Project Structure

The repository includes the following:

- **`Sentiment_Analysis_through_Twitter_Tweets.ipynb`**: Jupyter Notebook containing the entire workflow for data preprocessing, model training, and evaluation.
- **Data**:
  - Preprocessed Twitter data (not included in this repository if large; refer to the instructions below)
- **Model**: Trained model file (e.g., `trainedmodel.sav`) provided for reuse.
- **Results**: Insights and metrics from the model's evaluation.
- **Dependencies**: Python libraries required to execute the notebook.

---

## Features

- **Data Collection**: Collects tweets from Twitter using APIs like Tweepy or datasets such as Sentiment140.
- **Preprocessing**: Cleans the text by removing special characters, URLs, hashtags, and other unnecessary components.
- **Feature Extraction**: Converts textual data into numerical format using techniques like TF-IDF or word embeddings.
- **Model Training**: Builds and trains machine learning models such as Logistic Regression or Naive Bayes, or deep learning models like LSTMs.
- **Evaluation**: Measures the model's performance using metrics like accuracy, precision, recall, and F1-score.

---

## Requirements

The project uses the **Sentiment140 dataset** with 1.6 million tweets, sourced from Kaggle.  
You can access the dataset here: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).

### Steps to Get Started:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/sentiment-analysis-twitter.git
   cd sentiment-analysis-twitter
   ```

2. Install the required dependencies:

3. Download the dataset from Kaggle and place it in the `data/` folder.

4. Open and execute the Jupyter Notebook:
   ```bash
   jupyter notebook Sentiment_Analysis_through_Twitter_Tweets.ipynb
   ```

---

## Libraries Used

- **NumPy**: Used for handling numerical arrays during data preprocessing and performing mathematical computations such as converting text data into numerical formats.
- **Pandas**: Used to load and manipulate the Twitter dataset, including tasks like filtering, cleaning, and organizing tweet data for analysis.
- **NLTK (Natural Language Toolkit)**: Used for text preprocessing, including tokenization, stopword removal, stemming, lemmatization, and preparing textual data for model training.
- **Scikit-learn (skLearn)**: Used for feature extraction (e.g., TF-IDF Vectorizer), building machine learning models (e.g., Logistic Regression), and evaluating the model's performance using metrics such as accuracy score.

---

## Pickled Dataset

The dataset used in this project has been preprocessed and saved in a serialized format (`trainedmodel.sav`). This pickled file ensures faster loading and reusability. It contains cleaned and tokenized tweet data, ready for training and testing.

To load the pickled model, use the following Python code:
```python
import pickle

# Load the pickled model
with open('trainedmodel.sav', 'rb') as file:
    model = pickle.load(file)

# Example usage:
predictions = model.predict(new_data)
```

---

## Model and Results

- The trained machine learning model (`trainedmodel.sav`) provided in this repository can be used to make predictions on new tweet data without retraining.
- The model has been optimized for sentiment classification and evaluated using metrics like accuracy and F1-score.

---

## Future Improvements

- Experiment with advanced transformer models like BERT for improved accuracy.
- Build a user-friendly web interface to deploy the model for real-time sentiment analysis.
- Use a larger and more diverse dataset to improve the model's robustness.

---
