from flask import Flask, render_template, request
from inference import detector

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        result, updated_text = detector(text)
        return render_template('index.html', prediction=result, text = updated_text)


if __name__ == '__main__':
    app.run(debug=True)
