import joblib


classifier = joblib.load('ai_text_detector.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def detector(text):

    chunks = text.split(". ")
    total_count = len(chunks)
    ai_count = 0
    for sent in chunks:
        email_vec = vectorizer.transform([sent])
        pred = classifier.predict(email_vec)



        if pred[0] == 1:
            ai_count = ai_count+1
            text = text.replace(sent, f"""<span class="ai-text">{sent}</span>""")
        else:
            continue

    total_ai_percentage = ((ai_count/total_count)*100)

    #print(text)

    return f"{round(total_ai_percentage, 2)} % AI Generated", text

