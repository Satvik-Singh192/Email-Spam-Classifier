import tensorflow as tf
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, render_template
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask_cors import CORS
import nltk
nltk.download('stopwords')

model = tf.keras.models.load_model('spam_classifier_model.keras')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)
CORS(app) 

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_message(mssg):
    mssg = mssg.lower()
    mssg = re.sub(r'[^\w\s]', '', mssg)    
    mssg = ' '.join([stemmer.stem(word) for word in mssg.split() if word not in stop_words])
    return mssg

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        sentence = request.form['sentence']
        preprocessed_sentence = preprocess_message(sentence)
        sentence_tfidf = vectorizer.transform([preprocessed_sentence]).toarray()
        prediction = model.predict(sentence_tfidf)
        predicted_class = "Spam" if prediction[0] >= 0.5 else "Not Spam"
        return render_template('index.html', 
                               sentence=sentence, 
                               prediction=predicted_class, 
                               probability=round(float(prediction[0][0]), 4))
    return render_template('index.html', sentence=None)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data['sentence']
    preprocessed_sentence = preprocess_message(sentence)
    sentence_tfidf = vectorizer.transform([preprocessed_sentence]).toarray()
    prediction = model.predict(sentence_tfidf)
    predicted_class = "spam" if prediction[0] >= 0.5 else "Not Spam"
    return jsonify({
        'sentence': sentence,
        'prediction': predicted_class,
        'probability': float(prediction[0][0])
    })

if __name__ == '__main__':
    app.run(debug=True)
