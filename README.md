# 📊 Social-Media-Patterns-Through-Data-Insights

## 📌 Project Overview

**Unravelling Social Media Patterns Through Data Insights** is a Machine Learning and Natural Language Processing (NLP) based project that analyzes social media data to identify sentiment patterns, user engagement, and trending topics.

The project focuses on social media platforms such as Twitter, Instagram, and Facebook. It uses text preprocessing, TF-IDF feature extraction, BERT embeddings, and Machine Learning algorithms to classify sentiments and discover hidden patterns in social media interactions.

## 🎯 Objectives

- Analyze social media text data.
- Identify positive and negative sentiments.
- Understand user engagement through likes and retweets.
- Discover hidden topics and patterns using clustering.
- Analyze sentiment distribution across social media data.
- Visualize social media trends and engagement patterns.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning algorithms |
| NLTK | Text preprocessing |
| TensorFlow | Deep Learning |
| Transformers | BERT embeddings |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Flask | Web application integration |

## 🤖 Algorithms and Techniques

1. **TF-IDF (Term Frequency-Inverse Document Frequency):** Converts text data into numerical features.
2. **BERT Embeddings:** Generates contextual representations of text.
3. **Logistic Regression:** Classifies social media text into sentiment categories.
4. **Random Forest Classifier:** Performs sentiment classification using an ensemble of decision trees.
5. **Deep Learning:** Uses neural network layers to classify sentiment based on BERT embeddings.
6. **K-Means Clustering:** Groups similar social media posts into clusters.
7. **Latent Dirichlet Allocation (LDA):** Identifies hidden topics in text data.

## 📂 Project Structure
Unravelling-Social-Media-Patterns/
│
├── dataset.csv
│
├── unravelling_social_media_patterns.ipynb
│
├── model.pkl
├── vectorizer.pkl
│
├── app.py
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css
│
├── requirements.txt
│
└── README.md
## 📊 Dataset Description

The dataset contains social media information with the following attributes:
- Text – Social media post content.
- Sentiment – Sentiment associated with the post.
- Timestamp – Date and time of the post.
- User – User information.
- Platform – Social media platform.
- Hashtags – Hashtags used in posts.
- Retweets – Number of retweets.
- Likes – Number of likes.
- Country – Country information.
- Year, Month, Day, Hour – Time-related attributes.

The notebook loads a dataset containing **732 records and 15 columns**.

## ⚙️ Project Workflow

### 1. Data Collection
Load social media data from the CSV dataset using Pandas.

### 2. Data Preprocessing
- Convert text into lowercase.
- Remove URLs.
- Remove special characters and numbers.
- Remove English stopwords.
- Generate cleaned text.

### 3. Feature Extraction
- Apply TF-IDF vectorization.
- Generate BERT-based text embeddings.

### 4. Model Training
Train the following models:
- Logistic Regression
- Random Forest Classifier
- Deep Learning Classifier

### 5. Clustering and Topic Modelling
- Apply K-Means clustering.
- Use LDA for topic modelling.

### 6. Data Visualization
Generate visualizations to understand:
- Sentiment distribution.
- Cluster distribution.
- Likes based on sentiment.
- Retweets based on sentiment.
- Daily likes over time.

### 7. Prediction
The Flask integration loads the trained Logistic Regression model and TF-IDF vectorizer to predict sentiment from uploaded CSV data.

## 📈 Results and Analysis

The project provides insights into:

- Positive and negative sentiment classification.
- Social media engagement patterns.
- Distribution of sentiments.
- Similarity between social media posts.
- Hidden topics and discussion patterns.
- Daily engagement trends.

Model accuracy should be reported after executing the training and evaluation cells successfully.

## 💻 Installation and Execution

### Step 1: Clone the Repository
### Step 2: Navigate to the Project Folder
### Step 3: Install Required Libraries
### Step 4: Run the Jupyter Notebook
### Step 5: Run the Flask Application
## 🔮 Future Enhancements

- Integrate real-time social media data.
- Add support for multilingual sentiment analysis.
- Improve prediction accuracy using advanced NLP models.
- Develop an interactive analytics dashboard.
- Add more social media platforms.
- Generate automated reports based on sentiment trends.

## Conclusion

This project demonstrates how Machine Learning, Deep Learning, and NLP techniques can be used to analyze social media data. It helps identify sentiment patterns, understand user engagement, discover topics, and extract meaningful insights from social media interactions.
