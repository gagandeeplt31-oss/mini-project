"""
fertilizer_calc.py - Fertilizer & Soil Amendment Calculator for AgriSmart.

Compares user's soil nutrient parameters (N, P, K, pH) against ideal agronomic baselines
derived from crop_recommendation.csv. Generates exact deficit/surplus metrics and
specific, actionable amendment advice (fertilizer applications, withholding recommendations,
agricultural lime, or elemental sulfur).
"""

import pandas as pd
import numpy as np

# Cache for crop baseline means: {crop_lower: {'N': float, 'P': float, 'K': float, 'ph': float}}
_CROP_BASELINES = None


def load_crop_baselines(dataset_path="crop_recommendation.csv"):
    """
    Computes and caches mean N, P, K, and pH values for each crop in crop_recommendation.csv.
    """
    global _CROP_BASELINES
    if _CROP_BASELINES is not None:
        return _CROP_BASELINES
    
    try:
        df = pd.read_csv(dataset_path)
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        grouped = df.groupby('label')[['N', 'P', 'K', 'ph']].mean().round(2)
        _CROP_BASELINES = grouped.to_dict(orient='index')
        return _CROP_BASELINES
    except Exception as e:
        print(f"[Warning] Failed to load crop baselines from {dataset_path}: {e}")
        return {}


def get_crop_baseline(crop_name, dataset_path="crop_recommendation.csv"):
    """
    Returns ideal baseline dict {'N': float, 'P': float, 'K': float, 'ph': float} for a given crop.
    """
    baselines = load_crop_baselines(dataset_path)
    crop_clean = str(crop_name).strip().lower()
    if crop_clean in baselines:
        return baselines[crop_clean]
    
    # Fallback to general agronomic averages if crop not found
    return {'N': 50.0, 'P': 50.0, 'K': 50.0, 'ph': 6.5}


def calculate_soil_amendments(user_soil, crop_name, dataset_path="crop_recommendation.csv", tolerance=2.0, ph_tolerance=0.25):
    """
    Calculates exact deficit/surplus for N, P, K and pH against the crop's ideal baseline.
    
    Args:
        user_soil (dict or list): Either dict with keys 'N', 'P', 'K', 'ph' or list [N, P, K, temp, hum, ph, rain]
        crop_name (str): The recommended crop name (e.g., 'rice', 'coffee')
        dataset_path (str): Path to crop_recommendation.csv
        tolerance (float): Nutrient unit tolerance for optimal classification
        ph_tolerance (float): pH tolerance for optimal classification
        
    Returns:
        dict: Detailed treatment plan with nutrient metrics, deltas, and actionable advice.
    """
    # Normalize user inputs
    if isinstance(user_soil, (list, tuple)):
        # Expected standard index: 0:N, 1:P, 2:K, 5:ph
        u_n = float(user_soil[0])
        u_p = float(user_soil[1])
        u_k = float(user_soil[2])
        u_ph = float(user_soil[5]) if len(user_soil) > 5 else 6.5
    elif isinstance(user_soil, dict):
        u_n = float(user_soil.get('N', user_soil.get('n', 0)))
        u_p = float(user_soil.get('P', user_soil.get('p', 0)))
        u_k = float(user_soil.get('K', user_soil.get('k', 0)))
        u_ph = float(user_soil.get('ph', user_soil.get('pH', 6.5)))
    else:
        raise ValueError("user_soil must be a dict or list/tuple of soil values")

    crop_clean = str(crop_name).strip().lower()
    ideal = get_crop_baseline(crop_clean, dataset_path=dataset_path)
    
    ideal_n = ideal['N']
    ideal_p = ideal['P']
    ideal_k = ideal['K']
    ideal_ph = ideal['ph']
    
    # Calculate deltas (user - ideal)
    # Negative delta = deficit (shortage)
    # Positive delta = surplus
    delta_n = round(u_n - ideal_n, 1)
    delta_p = round(u_p - ideal_p, 1)
    delta_k = round(u_k - ideal_k, 1)
    delta_ph = round(u_ph - ideal_ph, 2)
    
    actionable_advice = []
    nutrients = {}
    
    # 1. NITROGEN (N)
    if delta_n < -tolerance:
        shortage = abs(delta_n)
        urea_equiv = round(shortage / 0.46, 1)  # Urea is ~46% Nitrogen
        status_n = "Deficit"
        advice_n = (
            f"**Nitrogen (N) Deficit:** Soil is short by **{shortage:.1f} kg/ha** compared to {crop_clean.capitalize()}'s "
            f"ideal ({ideal_n:.1f} kg/ha). Apply **{shortage:.1f} kg/ha of pure N** "
            f"(equivalent to approx. **{urea_equiv:.1f} kg/ha of Urea [46% N]**) split across early vegetative stages."
        )
    elif delta_n > tolerance:
        surplus = delta_n
        status_n = "Surplus"
        advice_n = (
            f"**Nitrogen (N) Surplus:** Soil has a surplus of **+{surplus:.1f} kg/ha** above the ideal ({ideal_n:.1f} kg/ha). "
            f"**Withhold nitrogen fertilizers** to prevent excessive vegetative foliar growth, delayed flowering, lodging, "
            f"and nitrate leaching toxicity."
        )
    else:
        status_n = "Optimal"
        advice_n = (
            f"**Nitrogen (N) Balanced:** Soil nitrogen ({u_n:.1f} kg/ha) is right on target for {crop_clean.capitalize()} "
            f"(ideal baseline: {ideal_n:.1f} kg/ha). Maintain standard maintenance application only."
        )
    
    nutrients['N'] = {
        'user': u_n,
        'ideal': ideal_n,
        'delta': delta_n,
        'status': status_n,
        'advice': advice_n
    }
    actionable_advice.append(advice_n)
    
    # 2. PHOSPHORUS (P)
    if delta_p < -tolerance:
        shortage = abs(delta_p)
        dap_equiv = round(shortage / 0.46, 1)  # DAP is ~46% P2O5
        status_p = "Deficit"
        advice_p = (
            f"**Phosphorus (P) Deficit:** Soil is short by **{shortage:.1f} kg/ha** compared to {crop_clean.capitalize()}'s "
            f"ideal ({ideal_p:.1f} kg/ha). Apply **{shortage:.1f} kg/ha of Phosphorus** "
            f"(e.g., via **{dap_equiv:.1f} kg/ha of DAP** or Single Super Phosphate [SSP]) as a basal dressing to promote strong root establishment."
        )
    elif delta_p > tolerance:
        surplus = delta_p
        status_p = "Surplus"
        advice_p = (
            f"**Phosphorus (P) Surplus:** Soil has a surplus of **+{surplus:.1f} kg/ha** above the ideal ({ideal_p:.1f} kg/ha). "
            f"**Withhold phosphate fertilizers** to prevent interference with zinc (Zn) and iron (Fe) micronutrient uptake."
        )
    else:
        status_p = "Optimal"
        advice_p = (
            f"**Phosphorus (P) Balanced:** Soil phosphorus ({u_p:.1f} kg/ha) matches the ideal requirement ({ideal_p:.1f} kg/ha) for {crop_clean.capitalize()}."
        )
        
    nutrients['P'] = {
        'user': u_p,
        'ideal': ideal_p,
        'delta': delta_p,
        'status': status_p,
        'advice': advice_p
    }
    actionable_advice.append(advice_p)
    
    # 3. POTASSIUM (K)
    if delta_k < -tolerance:
        shortage = abs(delta_k)
        mop_equiv = round(shortage / 0.60, 1)  # MOP is ~60% K2O
        status_k = "Deficit"
        advice_k = (
            f"**Potassium (K) Deficit:** Soil is short by **{shortage:.1f} kg/ha** compared to {crop_clean.capitalize()}'s "
            f"ideal ({ideal_k:.1f} kg/ha). Apply **{shortage:.1f} kg/ha of Potassium** "
            f"(equivalent to approx. **{mop_equiv:.1f} kg/ha of Muriate of Potash [MOP 60% K₂O]**) to bolster disease resistance, drought tolerance, and grain/fruit filling."
        )
    elif delta_k > tolerance:
        surplus = delta_k
        status_k = "Surplus"
        advice_k = (
            f"**Potassium (K) Surplus:** Soil has a surplus of **+{surplus:.1f} kg/ha** above the ideal ({ideal_k:.1f} kg/ha). "
            f"**Withhold potassium fertilizers** to avoid antagonizing magnesium (Mg) and calcium (Ca) absorption."
        )
    else:
        status_k = "Optimal"
        advice_k = (
            f"**Potassium (K) Balanced:** Soil potassium ({u_k:.1f} kg/ha) meets {crop_clean.capitalize()}'s optimal baseline ({ideal_k:.1f} kg/ha)."
        )
        
    nutrients['K'] = {
        'user': u_k,
        'ideal': ideal_k,
        'delta': delta_k,
        'status': status_k,
        'advice': advice_k
    }
    actionable_advice.append(advice_k)
    
    # 4. SOIL pH AMENDMENT
    if delta_ph < -ph_tolerance:
        status_ph = "Too Acidic"
        advice_ph = (
            f"**Soil pH is Too Acidic ({u_ph:.2f} vs Ideal {ideal_ph:.2f}):** Soil acidity impairs nutrient availability and microbial health. "
            f"Apply **agricultural lime** (calcium carbonate, CaCO₃) or **dolomitic limestone** to raise soil pH into {crop_clean.capitalize()}'s optimal zone."
        )
    elif delta_ph > ph_tolerance:
        status_ph = "Too Alkaline"
        advice_ph = (
            f"**Soil pH is Too Alkaline ({u_ph:.2f} vs Ideal {ideal_ph:.2f}):** High alkalinity reduces phosphorus and micronutrient solubility. "
            f"Apply **elemental sulfur** (or agricultural gypsum / organic compost) to gently lower soil pH into {crop_clean.capitalize()}'s ideal range."
        )
    else:
        status_ph = "Optimal"
        advice_ph = (
            f"**Soil pH is Optimal ({u_ph:.2f}):** Soil pH is well-balanced and within the ideal range ({ideal_ph:.2f}) for {crop_clean.capitalize()}. No pH amendment required."
        )
        
    ph_info = {
        'user': u_ph,
        'ideal': ideal_ph,
        'delta': delta_ph,
        'status': status_ph,
        'advice': advice_ph
    }
    actionable_advice.append(advice_ph)
    
    return {
        'crop': crop_clean,
        'nutrients': nutrients,
        'ph': ph_info,
        'actionable_advice': actionable_advice
    }


if __name__ == "__main__":
    print("--- Testing fertilizer_calc.py ---")
    sample_soil = [90, 42, 43, 20.8, 82.0, 6.5, 202.9]
    plan = calculate_soil_amendments(sample_soil, 'rice')
    print(f"Crop: {plan['crop']}")
    for k, v in plan['nutrients'].items():
        print(f"{k}: User={v['user']}, Ideal={v['ideal']}, Delta={v['delta']:+g}, Status={v['status']}")
    print(f"pH: User={plan['ph']['user']}, Ideal={plan['ph']['ideal']}, Delta={plan['ph']['delta']:+g}, Status={plan['ph']['status']}")
    print("\nActionable Advice:")
    for a in plan['actionable_advice']:
        print(" -", a)
