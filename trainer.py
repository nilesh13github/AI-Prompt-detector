import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
import joblib


data = pd.read_csv('AI_Human.csv') # downloaded from https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data

text=data['text']
label=data['generated'] #0 for ham and 1 for AI

x_train, x_test, y_train, y_test = train_test_split(text, label, test_size=0.2, 
                                                    random_state=36)

vectorizer = TfidfVectorizer(stop_words='english')

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

classifier = MultinomialNB()

classifier.fit(x_train_vec, y_train)

y_pred = classifier.predict(x_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"the accuracy score is : {accuracy*100}")

joblib.dump(classifier, "ai_text_detector.pkl")
joblib.dump(vectorizer, "vectorizer.pkl" )

print("model saved!")