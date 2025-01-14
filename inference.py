import joblib


classifier = joblib.load('ai_text_detector.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def detector(text):
    email_vec = vectorizer.transform([text])
    pred = classifier.predict(email_vec)

    if pred[0] == 1:
        return "This message is AI Generated"
    else:
        return "This message is Human Written"
