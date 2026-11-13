"""
roi_calculator.py - Farm Economics & Return on Investment (ROI) Calculator.
Calculates estimated fertilizer amendment costs, projected crop yield,
gross market revenue, and estimated net profit based on live market spot prices
and land size in hectares.
"""

# Standard average crop yield baselines (in kg per hectare) under good management
# Derived from ICAR (Indian Council of Agricultural Research) & Ministry of Agriculture statistics
CROP_YIELDS = {
    'apple': 14000.0,       # 14.0 tonnes/ha
    'banana': 35000.0,      # 35.0 tonnes/ha
    'blackgram': 800.0,     # 0.8 tonnes/ha
    'chickpea': 1100.0,     # 1.1 tonnes/ha
    'coconut': 10000.0,     # 10.0 tonnes/ha (fresh nut biomass equivalent)
    'coffee': 1200.0,       # 1.2 tonnes/ha (clean beans)
    'cotton': 2000.0,       # 2.0 tonnes/ha (seed cotton)
    'grapes': 22000.0,      # 22.0 tonnes/ha
    'jute': 2600.0,         # 2.6 tonnes/ha (baled raw fiber)
    'kidneybeans': 1200.0,  # 1.2 tonnes/ha
    'lentil': 1000.0,       # 1.0 tonnes/ha
    'maize': 3200.0,        # 3.2 tonnes/ha
    'mango': 8500.0,        # 8.5 tonnes/ha
    'mothbeans': 600.0,     # 0.6 tonnes/ha
    'mungbean': 800.0,      # 0.8 tonnes/ha
    'muskmelon': 18000.0,   # 18.0 tonnes/ha
    'orange': 12000.0,      # 12.0 tonnes/ha
    'papaya': 45000.0,      # 45.0 tonnes/ha
    'pigeonpeas': 1000.0,   # 1.0 tonnes/ha
    'pomegranate': 10000.0, # 10.0 tonnes/ha
    'rice': 3600.0,         # 3.6 tonnes/ha (paddy)
    'watermelon': 30000.0   # 30.0 tonnes/ha
}

# Standard Indian retail market prices for common chemical fertilizers (₹/kg)
# Rates based on Government of India subsidized maximum retail prices:
# - Urea (46% N): ₹266.50 / 45kg bag ≈ ₹5.92 - ₹6.00 / kg
# - DAP (18% N, 46% P2O5): ₹1,350 / 50kg bag ≈ ₹27.00 / kg
# - MOP (60% K2O): ₹1,700 / 50kg bag ≈ ₹34.00 / kg
DEFAULT_FERTILIZER_PRICES = {
    'urea': 6.0,  # ₹/kg (Urea 46% N)
    'dap': 27.0,  # ₹/kg (Di-Ammonium Phosphate 46% P2O5)
    'mop': 34.0   # ₹/kg (Muriate of Potash 60% K2O)
}


def calculate_farm_economics(crop_name: str,
                             market_price: float,
                             farm_size_ha: float = 1.0,
                             treatment_plan: dict = None,
                             custom_prices: dict = None) -> dict:
    """
    Computes farm economics, amendment costs, projected revenue, and net profit.

    Args:
        crop_name (str): The recommended crop name (e.g. 'rice', 'cotton').
        market_price (float): Live or historical market rate in ₹/kg.
        farm_size_ha (float): Farm cultivable size in hectares (defaults to 1.0).
        treatment_plan (dict): The dictionary returned by fertilizer_calc.calculate_soil_amendments.
        custom_prices (dict): Optional custom fertilizer rates in ₹/kg.

    Returns:
        dict: Detailed economics metrics, revenue, cost breakdown, and net profit.
    """
    clean_crop = str(crop_name).strip().lower()
    farm_ha = max(0.01, float(farm_size_ha))
    mkt_price = max(0.0, float(market_price))

    prices = dict(DEFAULT_FERTILIZER_PRICES)
    if custom_prices and isinstance(custom_prices, dict):
        prices.update(custom_prices)

    # 1. EXTRACT NUTRIENT DEFICITS
    deficit_n = 0.0
    deficit_p = 0.0
    deficit_k = 0.0

    if treatment_plan and isinstance(treatment_plan, dict):
        nutrients = treatment_plan.get('nutrients', {})
        delta_n = float(nutrients.get('N', {}).get('delta', 0.0))
        delta_p = float(nutrients.get('P', {}).get('delta', 0.0))
        delta_k = float(nutrients.get('K', {}).get('delta', 0.0))

        # Deficit is positive when delta (user - ideal) is negative
        if delta_n < 0:
            deficit_n = abs(delta_n)
        if delta_p < 0:
            deficit_p = abs(delta_p)
        if delta_k < 0:
            deficit_k = abs(delta_k)

    # 2. CALCULATE COMMERCIAL FERTILIZER REQUIREMENTS (kg)
    # Urea: 46% Nitrogen
    # DAP: 46% Phosphorus (P2O5)
    # MOP: 60% Potassium (K2O)
    urea_kg = (deficit_n / 0.46) * farm_ha
    dap_kg = (deficit_p / 0.46) * farm_ha
    mop_kg = (deficit_k / 0.60) * farm_ha

    urea_cost = urea_kg * prices['urea']
    dap_cost = dap_kg * prices['dap']
    mop_cost = mop_kg * prices['mop']

    total_fertilizer_cost = urea_cost + dap_cost + mop_cost

    # 3. PROJECTED YIELD & REVENUE
    baseline_yield_ha = CROP_YIELDS.get(clean_crop, 3000.0)
    total_yield_kg = baseline_yield_ha * farm_ha
    gross_revenue = total_yield_kg * mkt_price

    # 4. PROFITABILITY & ROI
    net_profit = gross_revenue - total_fertilizer_cost
    profit_margin_pct = (net_profit / gross_revenue * 100.0) if gross_revenue > 0 else 0.0
    roi_pct = ((net_profit / total_fertilizer_cost) * 100.0) if total_fertilizer_cost > 0 else 0.0

    return {
        'crop': clean_crop,
        'farm_size_ha': round(farm_ha, 2),
        'market_price_per_kg': round(mkt_price, 2),
        'yield_per_ha_kg': round(baseline_yield_ha, 1),
        'total_yield_kg': round(total_yield_kg, 1),
        'gross_revenue': round(gross_revenue, 2),
        'fertilizer_cost': round(total_fertilizer_cost, 2),
        'net_profit': round(net_profit, 2),
        'profit_margin_pct': round(profit_margin_pct, 1),
        'roi_pct': round(roi_pct, 1),
        'fertilizer_breakdown': {
            'Urea': {
                'nutrient': 'Nitrogen (N)',
                'deficit_per_ha': round(deficit_n, 1),
                'total_qty_kg': round(urea_kg, 1),
                'rate_per_kg': prices['urea'],
                'cost': round(urea_cost, 2)
            },
            'DAP': {
                'nutrient': 'Phosphorus (P)',
                'deficit_per_ha': round(deficit_p, 1),
                'total_qty_kg': round(dap_kg, 1),
                'rate_per_kg': prices['dap'],
                'cost': round(dap_cost, 2)
            },
            'MOP': {
                'nutrient': 'Potassium (K)',
                'deficit_per_ha': round(deficit_k, 1),
                'total_qty_kg': round(mop_kg, 1),
                'rate_per_kg': prices['mop'],
                'cost': round(mop_cost, 2)
            }
        }
    }


if __name__ == "__main__":
    print("Testing roi_calculator.py...")
    from fertilizer_calc import calculate_soil_amendments
    sample_soil = [60, 20, 25, 25.0, 70.0, 6.5, 120.0]
    t_plan = calculate_soil_amendments(sample_soil, 'rice')
    res = calculate_farm_economics(crop_name='rice', market_price=38.6, farm_size_ha=2.5, treatment_plan=t_plan)
    print(f"Crop: {res['crop'].upper()} | Farm Size: {res['farm_size_ha']} ha")
    print(f"Total Yield: {res['total_yield_kg']} kg @ ₹{res['market_price_per_kg']}/kg")
    print(f"Gross Revenue: ₹{res['gross_revenue']:,.2f}")
    print(f"Fertilizer Cost: ₹{res['fertilizer_cost']:,.2f}")
    print(f"Net Profit: ₹{res['net_profit']:,.2f} (Margin: {res['profit_margin_pct']}%)")
    for fert, d in res['fertilizer_breakdown'].items():
        print(f" - {fert}: {d['total_qty_kg']} kg @ ₹{d['rate_per_kg']}/kg = ₹{d['cost']:,.2f}")
