import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_preprocess():
    print("Step 1: Loading CSV file...")
    df = pd.read_csv('crop_recommendation.csv')
    
    # Force features and target into NumPy arrays immediately
    # This avoids the 'narwhals' metadata scan that freezes Python 3.13
    X = df.drop('label', axis=1).values 
    y = df['label'].values
    
    print(f"Successfully loaded {len(df)} rows of data.")
    print("Step 2: Splitting and Scaling data...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    # Ensure scaled data is explicitly a float64 numpy array
    X_train_scaled = np.array(scaler.fit_transform(X_train), dtype=np.float64)
    X_test_scaled = np.array(scaler.transform(X_test), dtype=np.float64)
    
    print("Data preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ==========================================
# 2. MODEL COMPARISON ENGINE
# ==========================================
def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}... please wait...")
        # Explicitly cast to numpy array again just to be safe
        model.fit(np.array(X_train), np.array(y_train))
        predictions = model.predict(np.array(X_test))
        
        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average='weighted'),
            "Recall": recall_score(y_test, predictions, average='weighted'),
            "F1-Score": f1_score(y_test, predictions, average='weighted')
        }
        results[name] = metrics
        print(f"Done with {name}!")

    return results

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Starting Stability-Enhanced ML Engine ---")
    try:
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
        comparison_results = compare_models(X_train, X_test, y_train, y_test)
        
        results_df = pd.DataFrame(comparison_results).T
        print("\n--- Model Comparison Table ---")
        print(results_df)
        
        results_df.to_csv('model_performance.csv')
        print("\nResults saved to model_performance.csv.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
