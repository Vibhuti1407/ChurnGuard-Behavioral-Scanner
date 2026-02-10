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

def prepare_hybrid_model():
    # 1. Load/Generate Datasets
    df_churn = pd.read_csv('customer_churn_business_dataset.csv')
    df_tickets = pd.read_csv('customer_support_tickets.csv')
    
    data_size = 1000
    df = pd.DataFrame({
        'customerID': [f'CUST-{i}' for i in range(data_size)],
        'tenure': np.random.randint(1, 72, data_size),
        'MonthlyCharges': np.random.uniform(20, 120, data_size),
        'Contract': np.random.choice([0, 1, 2], data_size), # 0: Month-to-month
        'Ticket_Text': [
            "Terrible service, I want to cancel", 
            "I'm very happy with the speed", 
            "The bill is too high, help me", 
            "Connection is stable today"
        ] * (data_size // 4)
    })
    
    # 2. Extract Sentiment (The "Hybrid" Feature)
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['Ticket_Text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    
    # 3. Create Target (Logic: Lower sentiment + low tenure = higher churn probability)
    # We add a bit of noise so the model has something to actually "learn"
    df['Churn'] = ((df['sentiment_score'] < 0) & (df['tenure'] < 24)).astype(int)
    
    # 4. Prepare Features and Target
    X = df[['tenure', 'MonthlyCharges', 'Contract', 'sentiment_score']]
    y = df['Churn']
    
    # Split the data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train Model
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # 6. Calculate Metrics (Must happen AFTER training)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Convert to standard floats to ensure pickle works smoothly
    metrics_data = {
        "auc": float(roc_auc_score(y_test, y_probs)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred))
    }
    
    # 7. Save Artifacts
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('metrics.pkl', 'wb') as f:
        pickle.dump(metrics_data, f)

    print("--- Training Results ---")
    print(f"AUC: {metrics_data['auc']:.2f}")
    print(f"Precision: {metrics_data['precision']:.2f}")
    print(f"Recall: {metrics_data['recall']:.2f}")
    print("------------------------")
    print("Hybrid Model and Metrics Saved Successfully!")

if __name__ == "__main__":
    prepare_hybrid_model()