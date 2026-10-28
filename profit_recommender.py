import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SETUP & MODEL TRAINING
# ==========================================
def train_winner_model():
    print("Loading data and training the winning model (Random Forest)...")
    try:
        df = pd.read_csv('crop_recommendation.csv')
        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        # Use the full dataset for maximum accuracy
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        print("Model trained successfully!")
        return model, scaler
    except FileNotFoundError:
        print("Error: 'crop_recommendation.csv' not found in the folder!")
        exit()

# ==========================================
# 2. THE SMART PROFIT LOGIC (Risk-Aware Version)
# ==========================================
def get_profitable_recommendation(user_data, model, scaler, market_df):
    # --- SETTINGS ---
    # Threshold: Crop must be at least 30% suitable to be considered for profit optimization.
    # This prevents recommending an expensive crop that will likely die.
    MIN_THRESHOLD = 0.30 
    
    # --- DATA CLEANING ---
    market_df.columns = market_df.columns.str.strip()
    market_df.columns = ['Crop', 'Price'] + list(market_df.columns[2:]) 
    market_df['Crop'] = market_df['Crop'].astype(str).str.strip().str.lower()
    
    # Scale user input
    user_scaled = scaler.transform([user_data])
    
    # Get probabilities for all crops
    probs = model.predict_proba(user_scaled)[0]
    classes = model.classes_
    
    # Get top 3 candidates biologically
    top_indices = np.argsort(probs)[-3:][::-1]
    top_candidates = [(classes[i], probs[i]) for i in top_indices]
    
    print("\n--- Biologically Suitable Crops ---")
    for crop, prob in top_candidates:
        print(f"{crop}: {prob*100:.2f}% suitability")
    
    # STEP 1: Filter out crops that are likely to fail (below threshold)
    viable_candidates = [c for c in top_candidates if c[1] >= MIN_THRESHOLD]
    
    best_crop = None
    highest_price = -1
    
    # STEP 2: If we have viable crops, optimize for profit
    if viable_candidates:
        print("\n--- Analyzing Market Prices for VIABLE Crops ---")
        for crop, prob in viable_candidates:
            crop_lower = crop.lower().strip()
            try:
                price_row = market_df[market_df['Crop'] == crop_lower]
                if not price_row.empty:
                    price = price_row['Price'].values[0]
                    print(f"{crop}: Market Price = ${price} (VIABLE)")
                    if price > highest_price:
                        highest_price = price
                        best_crop = crop
            except Exception as e:
                print(f"{crop}: Error reading price. {e}")
    else:
        # STEP 3: FALLBACK
        # If NO crop meets the 30% threshold, we ignore profit and just pick the most suitable one.
        print("\n--- Warning: No crops meet the suitability threshold ---")
        print("Falling back to the most biologically suitable crop to avoid total crop failure.")
        best_crop = top_candidates[0][0] 
        try:
            crop_lower = best_crop.lower().strip()
            highest_price = market_df[market_df['Crop'] == crop_lower]['Price'].values[0]
        except:
            highest_price = "Unknown"

    return best_crop, highest_price

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load market data
    try:
        market_df = pd.read_csv('market_prices.csv')
        print("Market prices loaded.")
    except FileNotFoundError:
        print("Error: 'market_prices.csv' not found!")
        exit()

    # 2. Train the winning Random Forest model
    winner_model, data_scaler = train_winner_model()
    
    # 3. SIMULATED USER INPUT 
    # Order: N, P, K, temp, humidity, ph, rainfall
    # Example: High N, Moderate P/K, Moderate Temp, High Humidity/Rainfall (Rice-like)
    sample_user_input = [90, 42, 43, 20.8, 82.0, 6.5, 202.9] 
    
    print("\nUser Input (N,P,K,Temp,Hum,pH,Rain):", sample_user_input)
    
    # 4. Get the final recommendation
    final_crop, final_price = get_profitable_recommendation(
        sample_user_input, winner_model, data_scaler, market_df
    )
    
    # 5. Final Output with Crash Protection
    if final_crop:
        print("\n" + "="*45)
        print(f"FINAL RECOMMENDATION: {final_crop.upper()}")
        print(f"Reason: Best balance of biological suitability and market value (${final_price}).")
        print("="*45)
