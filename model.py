import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def prepare_hybrid_model():
    # 1. Load/Generate Datasets
    df_churn = pd.read_csv('customer_churn_business_dataset.csv')
    df_tickets = pd.read_csv('customer_support_tickets.csv')
    
    data_size = 1000
    df = pd.DataFrame({
        'customerID': [f'CUST-{i}' for i in range(data_size)],
        'tenure': np.random.randint(1, 72, data_size),
        'MonthlyCharges': np.random.uniform(20, 120, data_size),
        'Contract': np.random.choice([0, 1, 2], data_size), 
        'Ticket_Text': ["Terrible service", "Great speed", "Too expensive", "Stable"] * (data_size // 4)
    })
    
    # 2. Advanced Feature Engineering
    df['sentiment_score'] = df['Ticket_Text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    
    # Define Ground Truth (Churn logic)
    df['Churn'] = (
        (df['MonthlyCharges'] > 95) | 
        (df['sentiment_score'] < -0.3) | 
        (df['Contract'] == 0) & (df['tenure'] < 10)
    ).astype(int)    

    # 3. Model Training
    X = df[['tenure', 'MonthlyCharges', 'Contract', 'sentiment_score']]
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # 4. Save Artifacts (Including X_train for SHAP)
    artifacts = {
        "model": model,
        "metrics": {
            "auc": float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])),
            "precision": float(precision_score(y_test, model.predict(X_test))),
            "recall": float(recall_score(y_test, model.predict(X_test)))
        },
        "X_train": X_train # Essential for SHAP explainability
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)

if __name__ == "__main__":
    prepare_hybrid_model()
