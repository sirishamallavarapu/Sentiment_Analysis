# Sentiment Analysis

## About the Project

This project is a **Sentiment Analysis system using Natural Language Processing (NLP) and Machine Learning**. It analyzes text data and classifies it into different sentiment categories.

The project also performs data analysis and visualization to understand sentiment patterns across platforms, countries, and other features.

## Features

* Data loading and exploration
* Missing value analysis
* Sentiment distribution analysis
* Platform-wise and country-wise sentiment analysis
* Text cleaning and preprocessing
* Stopword removal
* Word frequency analysis
* WordCloud generation
* TF-IDF feature extraction
* Machine Learning model training
* Model accuracy comparison
* Sentiment prediction for new text
* Saving the trained model and TF-IDF vectorizer

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **NLTK**
* **WordCloud**
* **Scikit-learn**
* **Joblib**

## Machine Learning Models

The project compares four Machine Learning algorithms:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear SVM
4. Random Forest

The model with the highest accuracy is selected as the best-performing model.

## NLP & Text Processing

The text is processed by:

* Converting text to lowercase
* Removing URLs
* Removing mentions and hashtags
* Removing special characters
* Removing stopwords
* Converting text into numerical features using **TF-IDF**

## Dataset

The project uses `sentimentdataset.csv`, which contains text and additional information such as:

* Sentiment
* Platform
* Country
* Likes
* Retweets
* Hour

A cleaned version of the dataset is also generated after preprocessing.

## Output

The project generates:

* Sentiment distribution graphs
* Platform and country sentiment visualizations
* Likes vs. retweets analysis
* Sentiment activity by hour
* Word frequency tables
* WordClouds
* Model accuracy comparison
* Classification report
* Best trained sentiment model

The trained model and TF-IDF vectorizer are saved using Joblib:

```text
best_sentiment_model.pkl
tfidf_vectorizer.pkl
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/sirishamallavarapu/Sentiment_Analysis.git
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn joblib
```

Run the project:

```bash
python sample1.py
```

## Project Structure

```text
Sentiment_Analysis/
│
├── sample1.py
├── sentimentdataset.csv
├── cleaned_sentiment_dataset.csv
├── best_sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── README.md
```

## Future Improvements

* Develop a web application for real-time sentiment prediction
* Improve the model using advanced NLP techniques
* Add deep learning models such as LSTM or BERT
* Deploy the project as an online application
