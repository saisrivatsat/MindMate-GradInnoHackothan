# train_model.py
import os
import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load dataset
df = pd.read_csv("/Users/saisrivatsat/Downloads/MindMate!/mental_statements.csv")
df = df.rename(columns={"statement": "text", "status": "label"})
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].apply(clean_text)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Create output folder
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# TF-IDF Vectorizer with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "MultinomialNB": MultinomialNB(),
    "LinearSVC": LinearSVC(class_weight="balanced", max_iter=2000),
    "RidgeClassifier": RidgeClassifier(class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced", max_depth=10),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Train, evaluate, and save each model
for name, model in models.items():
    print(f"\n Training: {name}")
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f" Evaluation Report for {name}:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join(model_dir, f"{name}_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f" Model saved as {model_path}")

print("\n All models trained and saved in the 'saved_models' directory.")
