from flask import Flask , request , render_template
import nltk
nltk.download('wordnet')
nltk.download('stpwords')
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

import pickle
tfidf = pickle.load(open("tfidf.pkl","rb"))
LR = pickle.load(open("LR_model.pkl","rb"))


app = Flask(__name__)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {"not", "no", "bad", "good"}

def cleaning_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Prediction', methods=['POST', 'GET'])
def Prediction():
    if request.method == "POST":
        email = request.form['Email']
        if email.strip():  # Check if not empty
            clean_email = cleaning_text(email)
            vectorize = tfidf.transform([clean_email])
            prediction = LR.predict(vectorize)[0]
            return render_template("index.html", prediction=prediction)
        else:
            return render_template("index.html", prediction="empty")
    else:
        return render_template("index.html")

    
if __name__=="__main__":
    app.run(debug=True)