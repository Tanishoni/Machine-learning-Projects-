from flask import Flask, request, render_template
#Flask : class for making object render_template : function to render HTML templates

import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle

import pickle

# Load the classifier model
with open('clf.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('tfidf.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)


# clf = pickle.load(open('clf.pkl', 'rb')) #load the model
# tfidf = pickle.load(open('tfidf.pkl', 'rb')) #load the mode/l

stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
# Removed the undefined "text" usage as it is not needed here


app = Flask(__name__)


def preprocess(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    text = re.sub(r'[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)

texts = [
    preprocess("I love this product"),
    preprocess("I hate this item"),
    preprocess("This is amazing"),
    preprocess("Terrible experience")
]
texts = [
    preprocess("I love this product"),
    preprocess("I hate this item"),
    preprocess("This is amazing"),
    preprocess("Terrible experience"),
    preprocess("Excellent quality"),
    preprocess("Worst thing I've ever bought"),
    preprocess("Superb and fast delivery"),
    preprocess("Not worth the money"),
    preprocess("Very satisfied with this"),
    preprocess("Broke after one day"),
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]



#@app.route : decorator to bind URL to function
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        #comment use for input text from user
        #request.form : to get data from HTML form
        comment = request.form['text']
        cleaned_comment = preprocess(comment)#need to transform the text to lower case and remove special characters
        comment_vector = tfidf.transform([cleaned_comment])#transform the text to vector using tfidf
    prediction =  clf.predict(comment_vector)[0]
    #predict the text using the model

    return render_template('index.html', prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)