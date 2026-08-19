import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from collections import Counter

import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

df = pd.read_csv("sentimentdataset.csv")

print(df.head())

#step 3
print("\n Dataset Information:")
df.info()

print("\n Missing Values:")
print(df.isnull().sum())

print("\n Numerical Summery:")
print(df.describe())

print("\n Sentiment Classes:")
print(df['Sentiment'].value_counts())

#step 4 Get top 10 setiments by count
top_sentiments = df['Sentiment'].value_counts().nlargest(10).index
df_top = df[df['Sentiment'].isin(top_sentiments)]

#4.1 Sentiment Distribution
plt.figure(figsize=(8,4))
sns.countplot(data=df_top, x='Sentiment', palette='coolwarm', order=top_sentiments)
plt.title('Top 10 Sentiment Distribution')
plt.xticks(rotation=45)
plt.show()

#4.2 Platform-wise Sentiment Distribution

plt.figure(figsize=(10,5))
sns.countplot(data=df_top, x='Platform', hue='Sentiment', palette='Set2')
plt.title('Platform-wise Sentiment (Top 10)')
plt.xticks(rotation=30)
plt.show()

#4.3 country-wise Sentiment
plt.figure(figsize=(12,5))
sns.countplot(data=df_top, x='Country', hue='Sentiment', palette='magma')
plt.title('Country-wise Sentiment Distribution (Top 10)')
plt.xticks(rotation=45)
plt.show()

#4.4 Likes vs Retweets
plt.figure(figsize=(8,4))
sns.scatterplot(data=df_top, x='Likes', y='Retweets', hue='Sentiment', alpha=0.7)
plt.title('Likes vs Retweets by Sentiment (Top 10)')
plt.show()

#4.5 Sentiment Trend over Hours
plt.figure(figsize=(10,4))
sns.countplot(data=df_top, x='Hour', hue='Sentiment',palette='cool')
plt.title('Sentiment Activity by Hour (Top 10)')
plt.show()

#step 5 : Text cleaning & Preprocessing
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) #remove URLs
    text = re.sub(r"@\w+", "", text) #remove mentions
    text = re.sub(r"#\w+", "", text) #remove hashtags
    text = re.sub(r"[^a-z\s]", "", text) # keep only letters
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()

df['Clean_Text'] = df['Text'].apply(clean_text)
df[['Text', 'Clean_Text']].head(10)

#step 6:Text Analysis

#Select top 10 most frequent sentiments
top_sentiments = df['Sentiment'].value_counts().nlargest(10).index
df_top = df[df['Sentiment'].isin(top_sentiments)]

# 6.1 Average Word Count per Sentiment (Top 10)
df_top['word_count'] = df_top['Clean_Text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10,5))
sns.boxplot(data=df_top, x='Sentiment', y='word_count', palette='viridis', order=top_sentiments) 
plt.title('Word Count Distribution by Top 10 Sentiments')
plt.xticks(rotation=45)
plt.show()

# 6.2 Most Common Words per Sentiment (Top 10 sentiments, top 5 words each)
def most_common_words(sentiment, n=5):
    words = " ".join(df_top[df_top['Sentiment']==sentiment]['Clean_Text']).split()
    common = Counter(words).most_common(n)
    return pd.DataFrame(common, columns=['Word', 'Frequency'])
# print(most_common_words(df[Sentiment])) 



for s in top_sentiments:
    print(f"\nTop words for {s}:")
    print(most_common_words(s, n=5)) # only top 5 words

# 6.3 WordClouds for Top 10 Sentiments
for sentiment in top_sentiments:
    text = " ".join(df_top[df_top['Sentiment']==sentiment]['Clean_Text'])
    wc = WordCloud(width=700, height=400, background_color='white', colormap="plasma", max_words=50).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud - {sentiment}")
    plt.show()

#step 7:Feature Extraction
#Combine rare classes with <2 samples into 'Other'
counts = df['Sentiment'].value_counts()
rare_classes = counts[counts < 2].index
df['Sentiment'] = df['Sentiment'].replace(rare_classes, 'Other')

#TF-IDF
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.95
)
X = tfidf.fit_transform(df['Clean_Text'])
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

    #step 9:Visual Comarision of Model Accuracies

    plt.figure(figsize=(7,4))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette='cubehelix')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.show()

    best_model_name = max(results, key=results.get)
    print(f"\n Best Performing Model: {best_model_name} ({results[best_model_name]:.2%} accuracy)")

    #step 10:Detailed Evaluation of Best model
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best))


    #step 11:Save Best Model and Vectorizer
    joblib.dump(best_model, "best_sentiment_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("\n Model and TF-IDF Vectorizer saved successfully!")

    #step 12: Predict Sentiment on New test
    def predict_sentiment(text):
        clean = clean_text(text)
        vec = tfidf.transform([clean])
        pred = best_model.predict(vec)[0]
        return pred
    #Example predictions
    samples = [
        "I love this new feature!",
        "The service was awful and slow."
        "It's okay, not too bad but not great either."
    ]

    for s in samples:
        print(f"Text: {s} -> Sentiment: {predict_sentiment(s)} ")

    #step 13: Export cleaned Dataset

    df.to_csv("cleaned_sentiment_dataset.csv", index=False)
    print("Cleaned dataset saved successfully!")