import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    
    text = text.lower()
    
    text = nltk.word_tokenize(text)
    
    text = [i for i in text if i.isalnum()]
    
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)

def prepare_data():
   
    df = pd.read_csv('spam.csv')
    
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    
    
    df.columns = ['label', 'text'] + list(df.columns[2:])
    
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    df = df.drop_duplicates(keep='first')
    
    df['transformed_text'] = df['text'].apply(transform_text)
    
    return df


def train_model(df):
   
    X = df['transformed_text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    nb_model = MultinomialNB()
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    
    models = {
        'Random Forest': rf_model,
        'Naive Bayes': nb_model,
        'Logistic Regression': lr_model
    }
    
    results = {}
    print("\n" + "="*50)
    print("📊 MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    for name, model in models.items():
        
        model.fit(X_train_tfidf, y_train)
        
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        results[name] = {'accuracy': accuracy, 'precision': precision}
    
        print(f"\n🤖 {name}")
        print("─"*30)
        print(f"📈 Accuracy:  {accuracy:.4f}")
        print(f"🎯 Precision: {precision:.4f}")
    
    
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_accuracy = results[best_model_name]['accuracy']
    best_precision = results[best_model_name]['precision']
    
    print("\n" + "="*50)
    print("🏆 BEST PERFORMING MODEL")
    print("="*50)
    print(f"📌 Model: {best_model_name}")
    print(f"📊 Performance Metrics:")
    print(f"   ├─ Accuracy:  {best_accuracy:.4f}")
    print(f"   └─ Precision: {best_precision:.4f}")
    print("\n✨ This model has been selected as the primary classifier!")
    print("="*50)
    
   
    print("\n📥 Saving models...")
    for name, model in models.items():
        filename = f'{name.lower().replace(" ", "_")}_model.pkl'
        with open(filename, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"   ✓ Saved {filename}")
    
    
    print("\n📥 Saving vectorizer...")
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(tfidf, vectorizer_file)
    print("   ✓ Saved vectorizer.pkl")
    
    return models, tfidf

if __name__ == "__main__":
   
    df = prepare_data()
    
    model, vectorizer = train_model(df)