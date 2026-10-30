from flask import Flask, render_template, request
import pandas as pd
import re
import pickle

app = Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))


# Load Dataset
df = pd.read_csv("dataset.csv")

def clean_text(text):
    return str(text).lower().strip()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input')
def input_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('message', '').strip()
    cleaned_input = clean_text(user_input)
    
    # Lookup in CSV
    df['match_text'] = df['Text'].apply(clean_text)
    match = df[df['match_text'] == cleaned_input]

    if not match.empty:
        row = match.iloc[0]
        res = {
            "sentiment": row['Sentiment'],
            "platform": row.get('Platform', 'N/A'),
            "likes": row.get('Likes', 0),
            "user": row.get('User', 'N/A'),
            "hashtag": row.get('Hashtags', '#None')
        }
    else:
        res = {"sentiment": "Unknown", "platform": "N/A", "likes": "N/A", "user": "N/A", "hashtag": "N/A"}

    return render_template('results.html', message=user_input, res=res)

if __name__ == '__main__':
    app.run(debug=True)