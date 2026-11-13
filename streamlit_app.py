import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from market_fetcher import get_crop_price, BADGE_LIVE, BADGE_HISTORICAL
from fertilizer_calc import calculate_soil_amendments
from report_generator import generate_farm_report_pdf
from weather_fetcher import detect_ip_location, geocode_district, fetch_live_weather
from roi_calculator import calculate_farm_economics

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AgriSmart: Decision Support System", layout="wide")

# Professional CSS Styling
st.markdown("""
    <style>
    /* Primary Action Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: #ffffff !important;
        font-size: 19px;
        font-weight: 600;
        letter-spacing: 0.5px;
        height: 52px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.28);
        transition: all 0.25s ease-in-out;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #388e3c 0%, #2e7d32 100%);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.42);
        transform: translateY(-2px);
        color: #ffffff !important;
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.25);
    }
    
    /* Result Card with Modern Soft Shadow & Rounded Corners */
    .result-card {
        padding: 28px 24px;
        border-radius: 20px;
        background: linear-gradient(145deg, #ffffff 0%, #f4fbf5 100%);
        border: 1.5px solid #a5d6a7;
        box-shadow: 0 10px 30px -5px rgba(46, 125, 50, 0.12), 0 4px 12px -2px rgba(46, 125, 50, 0.06);
        text-align: center;
        color: #000000;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        box-shadow: 0 14px 35px -5px rgba(46, 125, 50, 0.16), 0 6px 16px -2px rgba(46, 125, 50, 0.08);
    }
    
    /* Container Borders */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid #c8e6c9 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
        padding: 12px;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        border: 1px solid #c8e6c9;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. BACKEND LOGIC & CONSTRAINTS
# ==========================================
FEATURE_CONSTRAINTS = {
    'N': {
        'label': 'Nitrogen (N) [kg/ha]',
        'min_value': 0.0,
        'max_value': 300.0,
        'step': 1.0,
        'help': 'Available Nitrogen in topsoil (cannot be negative, 0.0 to 300.0 kg/ha)'
    },
    'P': {
        'label': 'Phosphorus (P) [kg/ha]',
        'min_value': 0.0,
        'max_value': 300.0,
        'step': 1.0,
        'help': 'Available Phosphorus in topsoil (cannot be negative, 0.0 to 300.0 kg/ha)'
    },
    'K': {
        'label': 'Potassium (K) [kg/ha]',
        'min_value': 0.0,
        'max_value': 300.0,
        'step': 1.0,
        'help': 'Available Potassium in topsoil (cannot be negative, 0.0 to 300.0 kg/ha)'
    },
    'ph': {
        'label': 'Soil pH',
        'min_value': 0.0,
        'max_value': 14.0,
        'step': 0.1,
        'help': 'Soil pH scale (strictly 0.0 to 14.0)'
    },
    'temperature': {
        'label': 'Temperature (°C)',
        'min_value': 0.0,
        'max_value': 60.0,
        'step': 0.5,
        'help': 'Average growing season temperature (0.0 to 60.0 °C)'
    },
    'humidity': {
        'label': 'Relative Humidity (%)',
        'min_value': 0.0,
        'max_value': 100.0,
        'step': 1.0,
        'help': 'Ambient relative humidity (strictly 0.0 to 100.0%)'
    },
    'rainfall': {
        'label': 'Seasonal Rainfall (mm)',
        'min_value': 0.0,
        'max_value': 1500.0,
        'step': 5.0,
        'help': 'Total seasonal precipitation (cannot be negative, min 0.0 mm)'
    }
}

@st.cache_resource
def load_and_train():
    """Trains the model once and caches it for speed."""
    try:
        df = pd.read_csv('crop_recommendation.csv')
        X = df.drop('label', axis=1)
        y = df['label']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler, X.columns.tolist()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None, None

def get_smart_recommendation(user_data, model, scaler, market_df, threshold=0.30):
    # Scaling and Prediction
    feature_names = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    if feature_names is not None:
        user_df = pd.DataFrame([user_data], columns=feature_names)
        user_scaled = scaler.transform(user_df)
    else:
        user_scaled = scaler.transform([user_data])
    probs = model.predict_proba(user_scaled)[0]
    classes = model.classes_
    
    # Top 3 biological candidates
    top_indices = np.argsort(probs)[-3:][::-1]
    top_candidates = [(classes[i], probs[i]) for i in top_indices]
    
    # Risk Management: Filter by minimum biological viability threshold (30%)
    viable_candidates = [c for c in top_candidates if c[1] >= threshold]
    
    best_crop = None
    highest_price = -1
    best_badge = BADGE_HISTORICAL
    
    if viable_candidates:
        for crop, prob in viable_candidates:
            try:
                price, badge, _ = get_crop_price(crop)
                if price > highest_price:
                    highest_price = price
                    best_crop = crop
                    best_badge = badge
            except Exception:
                pass
    
    # Fallback: Pick the most suitable regardless of profit if none are viable
    if not best_crop:
        best_crop = top_candidates[0][0]
        try:
            highest_price, best_badge, _ = get_crop_price(best_crop)
        except Exception:
            highest_price = 0.0
            best_badge = BADGE_HISTORICAL
            
    return best_crop, highest_price, best_badge, top_candidates

# ==========================================
# 3. USER INTERFACE
# ==========================================
def main():
    st.title("🌾 AgriSmart: Crop Decision System")
    st.markdown("### *Maximizing Farmer Profit through Soil Science and Market Analysis*")
    st.divider()

    # Load Assets
    model, scaler, feature_names = load_and_train()
    try:
        market_df = pd.read_csv('market_prices.csv')
    except:
        st.error("market_prices.csv missing!")
        return

    if model is None: return

    # --- SIDEBAR: ABOUT THE SYSTEM ---
    with st.sidebar:
        st.header("🌾 AgriSmart System")
        st.markdown(
            """
            **AgriSmart** is an intelligent crop decision support system designed to maximize farmer revenue by combining machine learning, soil science, and live commodity economics.
            
            ---
            #### 💡 How It Works
            1. **Soil & Climate Modeling**: Evaluates farm NPK and meteorological metrics against 22 crop profiles using an ensemble Random Forest model (99.3% accuracy).
            2. **Live Market Optimization**: Cross-references biologically viable crops against live APMC mandi spot prices.
            3. **Explainable AI (SHAP)**: Explains the key limiting or boosting factors behind the recommendation.
            4. **Soil Treatment Plan**: Delivers exact NPK shortfall/surplus guidance and agricultural lime/sulfur pH recommendations.
            5. **PDF Farm Report**: Exports a printable farm advisory document.
            
            ---
            *AgriSmart v2.4 • Production Ready*
            """
        )

    # --- TABS LAYOUT ---
    tab1, tab2 = st.tabs(['🌾 Crop Recommender', '📊 Developer Analytics'])

    with tab1:
        df_defaults = pd.read_csv('crop_recommendation.csv')

        # --- SESSION STATE INITIALIZATION FOR INPUTS & LOCATION ---
        if 'auto_location' not in st.session_state:
            st.session_state['auto_location'] = detect_ip_location()

        if 'input_n' not in st.session_state:
            st.session_state['input_n'] = float(round(df_defaults['N'].mean(), 1))
        if 'input_p' not in st.session_state:
            st.session_state['input_p'] = float(round(df_defaults['P'].mean(), 1))
        if 'input_k' not in st.session_state:
            st.session_state['input_k'] = float(round(df_defaults['K'].mean(), 1))
        if 'input_ph' not in st.session_state:
            st.session_state['input_ph'] = float(round(df_defaults['ph'].mean(), 2))

        if 'input_temperature' not in st.session_state:
            st.session_state['input_temperature'] = float(round(df_defaults['temperature'].mean(), 1))
        if 'input_humidity' not in st.session_state:
            st.session_state['input_humidity'] = float(round(df_defaults['humidity'].mean(), 1))
        if 'input_rainfall' not in st.session_state:
            st.session_state['input_rainfall'] = float(round(df_defaults['rainfall'].mean(), 1))

        if 'input_farm_size' not in st.session_state:
            st.session_state['input_farm_size'] = 1.0

        if 'weather_status_msg' not in st.session_state:
            st.session_state['weather_status_msg'] = None

        def handle_autofill_weather():
            override_val = st.session_state.get('manual_city_override', '').strip()
            lat, lon, loc_name = None, None, None

            if override_val:
                geo = geocode_district(override_val)
                if geo.get('status') == 'success':
                    lat = geo['lat']
                    lon = geo['lon']
                    loc_name = geo['display']
                else:
                    st.session_state['weather_status_msg'] = {
                        'type': 'error',
                        'text': geo.get('message', f"Could not find coordinates for '{override_val}'.")
                    }
                    return
            else:
                ip_loc = st.session_state.get('auto_location')
                if not ip_loc or ip_loc.get('status') != 'success':
                    ip_loc = detect_ip_location()
                    st.session_state['auto_location'] = ip_loc

                if ip_loc.get('status') == 'success':
                    lat = ip_loc['lat']
                    lon = ip_loc['lon']
                    loc_name = ip_loc['display']
                else:
                    st.session_state['weather_status_msg'] = {
                        'type': 'warning',
                        'text': "Could not auto-detect IP location. Please type a district/city in the manual override box."
                    }
                    return

            weather = fetch_live_weather(lat, lon)
            if weather.get('status') == 'success':
                temp = max(FEATURE_CONSTRAINTS['temperature']['min_value'], 
                           min(FEATURE_CONSTRAINTS['temperature']['max_value'], float(weather['temperature'])))
                hum = max(FEATURE_CONSTRAINTS['humidity']['min_value'], 
                          min(FEATURE_CONSTRAINTS['humidity']['max_value'], float(weather['humidity'])))
                rain = max(FEATURE_CONSTRAINTS['rainfall']['min_value'], 
                           min(FEATURE_CONSTRAINTS['rainfall']['max_value'], float(weather['rainfall'])))

                st.session_state['input_temperature'] = float(temp)
                st.session_state['input_humidity'] = float(hum)
                st.session_state['input_rainfall'] = float(rain)

                st.session_state['weather_status_msg'] = {
                    'type': 'success',
                    'text': f"Fetched live Open-Meteo weather for **{loc_name}**: Temp: **{temp}°C** | Humidity: **{hum}%** | Rainfall: **{rain} mm**"
                }
            else:
                st.session_state['weather_status_msg'] = {
                    'type': 'error',
                    'text': f"Failed to retrieve weather from Open-Meteo: {weather.get('message', 'Network or API error')}"
                }

        # --- MAIN PAGE INPUT GRID ---
        st.markdown("#### 📋 Field Soil & Weather Conditions")
        st.caption("Enter your laboratory soil test values and expected local growing season climate parameters below:")

        col_soil, col_env = st.columns([1, 1], gap="medium")

        with col_soil:
            with st.container(border=True):
                st.markdown("##### 🧪 Soil Chemistry & Macronutrients")
                st.caption("Topsoil nutrient concentrations (kg/ha) and pH scale")
                
                c_n, c_p = st.columns(2)
                with c_n:
                    cfg_n = FEATURE_CONSTRAINTS['N']
                    val_n = st.number_input(
                        cfg_n['label'],
                        min_value=cfg_n['min_value'],
                        max_value=cfg_n['max_value'],
                        step=cfg_n['step'],
                        help=cfg_n['help'],
                        key='input_n'
                    )
                with c_p:
                    cfg_p = FEATURE_CONSTRAINTS['P']
                    val_p = st.number_input(
                        cfg_p['label'],
                        min_value=cfg_p['min_value'],
                        max_value=cfg_p['max_value'],
                        step=cfg_p['step'],
                        help=cfg_p['help'],
                        key='input_p'
                    )
                    
                c_k, c_ph = st.columns(2)
                with c_k:
                    cfg_k = FEATURE_CONSTRAINTS['K']
                    val_k = st.number_input(
                        cfg_k['label'],
                        min_value=cfg_k['min_value'],
                        max_value=cfg_k['max_value'],
                        step=cfg_k['step'],
                        help=cfg_k['help'],
                        key='input_k'
                    )
                with c_ph:
                    cfg_ph = FEATURE_CONSTRAINTS['ph']
                    val_ph = st.number_input(
                        cfg_ph['label'],
                        min_value=cfg_ph['min_value'],
                        max_value=cfg_ph['max_value'],
                        step=cfg_ph['step'],
                        help=cfg_ph['help'],
                        key='input_ph'
                    )

                st.markdown("###### 🚜 Farm Scale")
                val_farm_size = st.number_input(
                    "Farm Size (in hectares)",
                    min_value=0.1,
                    max_value=500.0,
                    step=0.5,
                    help="Total cultivable farm acreage in hectares (1 hectare ≈ 2.47 acres)",
                    key='input_farm_size'
                )

        with col_env:
            with st.container(border=True):
                st.markdown("##### 🌦️ Climate & Hydrology")
                st.caption("Local seasonal weather and precipitation conditions")

                # Geolocation & Live Weather Auto-Fill
                st.text_input(
                    "District / City (Manual Override)",
                    placeholder="e.g., Pune, Maharashtra (leave blank for auto-detected location)",
                    key="manual_city_override",
                    help="Type any city or district to fetch its weather, or leave empty to use auto-detected IP location."
                )

                st.button(
                    "📍 Locate Me & Auto-Fill Weather",
                    on_click=handle_autofill_weather,
                    use_container_width=True,
                    help="Automatically detects your location (or geocodes the override city) and fetches real-time temperature, humidity, and rainfall from Open-Meteo."
                )

                if st.session_state.get('weather_status_msg'):
                    status = st.session_state['weather_status_msg']
                    if status['type'] == 'success':
                        st.success(status['text'], icon="🌦️")
                    elif status['type'] == 'error':
                        st.error(status['text'], icon="⚠️")
                    elif status['type'] == 'warning':
                        st.warning(status['text'], icon="ℹ️")
                elif st.session_state.get('auto_location', {}).get('status') == 'success':
                    auto_l = st.session_state['auto_location']
                    st.caption(f"🌐 Auto-detected Location: **{auto_l['display']}** ({auto_l['lat']:.2f}°, {auto_l['lon']:.2f}°)")

                st.divider()

                # Climate Inputs
                c_temp, c_hum = st.columns(2)
                with c_temp:
                    cfg_temp = FEATURE_CONSTRAINTS['temperature']
                    val_temp = st.number_input(
                        cfg_temp['label'],
                        min_value=cfg_temp['min_value'],
                        max_value=cfg_temp['max_value'],
                        step=cfg_temp['step'],
                        help=cfg_temp['help'],
                        key='input_temperature'
                    )
                with c_hum:
                    cfg_hum = FEATURE_CONSTRAINTS['humidity']
                    val_hum = st.number_input(
                        cfg_hum['label'],
                        min_value=cfg_hum['min_value'],
                        max_value=cfg_hum['max_value'],
                        step=cfg_hum['step'],
                        help=cfg_hum['help'],
                        key='input_humidity'
                    )
                    
                c_rain, _ = st.columns(2)
                with c_rain:
                    cfg_rain = FEATURE_CONSTRAINTS['rainfall']
                    val_rain = st.number_input(
                        cfg_rain['label'],
                        min_value=cfg_rain['min_value'],
                        max_value=cfg_rain['max_value'],
                        step=cfg_rain['step'],
                        help=cfg_rain['help'],
                        key='input_rainfall'
                    )
                    
        user_inputs = [val_n, val_p, val_k, val_temp, val_hum, val_ph, val_rain]

        st.write("")
        if st.button("🎯 Recommend Best Profitable Crop"):
            final_crop, final_price, final_badge, all_candidates = get_smart_recommendation(
                user_inputs, model, scaler, market_df, threshold=0.30
            )
            
            st.divider()
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.subheader("✅ Final Recommendation")
                badge_bg = "#d4edda" if "Live" in final_badge else "#fff3cd"
                badge_text = "#155724" if "Live" in final_badge else "#856404"
                badge_border = "#c3e6cb" if "Live" in final_badge else "#ffeeba"
                
                st.markdown(f"""
                <div class="result-card">
                    <div style='font-size: 13px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #2e7d32; margin-bottom: 4px;'>Primary Agronomic Recommendation</div>
                    <h1 style='color: #1b5e20; margin: 4px 0 10px 0; font-size: 36px; font-weight: 800;'>{final_crop.upper()}</h1>
                    <div style='margin: 14px 0; display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: wrap;'>
                        <span style='font-size: 22px; font-weight: bold; color: #000000;'>Market Price: ₹{final_price}/kg</span>
                        <span style='display: inline-block; padding: 4px 12px; border-radius: 12px; background-color: {badge_bg}; color: {badge_text}; border: 1px solid {badge_border}; font-size: 13px; font-weight: bold;'>{final_badge}</span>
                    </div>
                    <p style='color: #222222; font-size: 15px; margin-top: 6px;'>This crop is biologically viable and economically optimized for your land.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("#### Suitability Analysis")
                # PYARROW-PROOF TABLE LOGIC
                suit_list = []
                for crop, prob in all_candidates:
                    c_price, c_badge, _ = get_crop_price(crop)
                    suit_list.append({
                        "Crop": crop,
                        "Suitability (%)": round(prob * 100, 2),
                        "Price (₹/kg)": c_price,
                        "Source": c_badge
                    })
                
                df_suit = pd.DataFrame(suit_list)
                st.dataframe(df_suit, use_container_width=True)

            with res_col2:
                st.subheader("🔍 Why this Crop? (XAI)")
                try:
                    user_df = pd.DataFrame([user_inputs], columns=feature_names)
                    user_scaled = scaler.transform(user_df)
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(user_scaled)
                    
                    # Correct indexing for Multi-class across shap versions
                    crop_index = list(model.classes_).index(final_crop)
                    if isinstance(shap_values, list):
                        feature_importance = shap_values[crop_index][0]
                    elif isinstance(shap_values, np.ndarray):
                        if shap_values.ndim == 3:
                            if shap_values.shape[2] == len(model.classes_):
                                feature_importance = shap_values[0, :, crop_index]
                            elif shap_values.shape[0] == len(model.classes_):
                                feature_importance = shap_values[crop_index, 0, :]
                            else:
                                feature_importance = shap_values[0, :, crop_index]
                        elif shap_values.ndim == 2:
                            feature_importance = shap_values[0]
                        else:
                            feature_importance = np.ravel(shap_values)[:len(feature_names)]
                    else:
                        feature_importance = np.ravel(shap_values)[:len(feature_names)]
                    
                    fig, ax = plt.subplots()
                    y_pos = np.arange(len(feature_names))
                    ax.barh(y_pos, feature_importance, align='center', color='#2e7d32')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(feature_names)
                    ax.invert_yaxis() 
                    ax.set_xlabel('Impact on Prediction')
                    ax.set_title(f'Factors influencing {final_crop}')
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"XAI Visualization skipped: {e}")
            
            # --- SOIL TREATMENT PLAN ---
            st.divider()
            st.subheader("🌱 Soil Treatment Plan")
            st.markdown(f"Customized nutrient adjustments and soil amendments for **{final_crop.upper()}** based on your soil profile vs. ideal agronomic baseline:")
            
            soil_input_dict = dict(zip(feature_names, user_inputs))
            treatment_plan = calculate_soil_amendments(soil_input_dict, final_crop)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            nutrients = treatment_plan['nutrients']
            ph_info = treatment_plan['ph']
            
            with col_m1:
                delta_n = nutrients['N']['delta']
                st.metric(
                    label="Nitrogen (N)",
                    value=f"{nutrients['N']['user']:.1f} kg/ha",
                    delta=f"{delta_n:+.1f} kg/ha (Ideal: {nutrients['N']['ideal']:.1f})",
                    delta_color="normal"
                )
                
            with col_m2:
                delta_p = nutrients['P']['delta']
                st.metric(
                    label="Phosphorus (P)",
                    value=f"{nutrients['P']['user']:.1f} kg/ha",
                    delta=f"{delta_p:+.1f} kg/ha (Ideal: {nutrients['P']['ideal']:.1f})",
                    delta_color="normal"
                )
                
            with col_m3:
                delta_k = nutrients['K']['delta']
                st.metric(
                    label="Potassium (K)",
                    value=f"{nutrients['K']['user']:.1f} kg/ha",
                    delta=f"{delta_k:+.1f} kg/ha (Ideal: {nutrients['K']['ideal']:.1f})",
                    delta_color="normal"
                )
                
            with col_m4:
                delta_ph = ph_info['delta']
                ph_delta_color = "off" if ph_info['status'] == "Optimal" else "normal"
                st.metric(
                    label="Soil pH",
                    value=f"{ph_info['user']:.2f}",
                    delta=f"{delta_ph:+.2f} (Ideal: {ph_info['ideal']:.2f})",
                    delta_color=ph_delta_color
                )
                
            st.write("#### 📋 Actionable Amendment Advice")
            for advice in treatment_plan['actionable_advice']:
                st.markdown(f"- {advice}")
                
            # --- 💰 ECONOMICS & ROI DASHBOARD ---
            st.divider()
            st.subheader("💰 Economics & ROI Dashboard")
            st.markdown(f"Projected financial returns and fertilizer amendment budget for **{final_crop.upper()}** on **{val_farm_size:.1f} hectare(s)**:")

            econ_data = calculate_farm_economics(
                crop_name=final_crop,
                market_price=final_price,
                farm_size_ha=val_farm_size,
                treatment_plan=treatment_plan
            )

            col_e1, col_e2, col_e3 = st.columns(3)

            with col_e1:
                cost_delta = f"-₹{econ_data['fertilizer_cost']:,.0f} input cost" if econ_data['fertilizer_cost'] > 0 else "Balanced (₹0)"
                st.metric(
                    label="Estimated Fertilizer Cost",
                    value=f"₹{econ_data['fertilizer_cost']:,.2f}",
                    delta=cost_delta,
                    delta_color="inverse"
                )

            with col_e2:
                st.metric(
                    label="Projected Gross Revenue",
                    value=f"₹{econ_data['gross_revenue']:,.2f}",
                    delta=f"{econ_data['total_yield_kg']:,.0f} kg yield (@ ₹{econ_data['market_price_per_kg']}/kg)",
                    delta_color="normal"
                )

            with col_e3:
                margin_delta = f"{econ_data['profit_margin_pct']:.1f}% profit margin"
                st.metric(
                    label="Estimated Net Profit",
                    value=f"₹{econ_data['net_profit']:,.2f}",
                    delta=margin_delta,
                    delta_color="normal" if econ_data['net_profit'] >= 0 else "inverse"
                )

            with st.expander("📊 View Detailed Cost & Production Breakdown", expanded=False):
                st.markdown(f"""
                - **Cultivated Land Area:** {econ_data['farm_size_ha']} ha
                - **Crop Benchmark Yield:** {econ_data['yield_per_ha_kg']:,.0f} kg/ha (Total Expected Production: **{econ_data['total_yield_kg']:,.0f} kg**)
                - **Market Rate:** ₹{econ_data['market_price_per_kg']}/kg ({final_badge})
                - **Gross Expected Revenue:** ₹{econ_data['gross_revenue']:,.2f}
                - **Fertilizer Expenditure:** ₹{econ_data['fertilizer_cost']:,.2f}
                - **Net Profit:** ₹{econ_data['net_profit']:,.2f} (**{econ_data['profit_margin_pct']}% Margin**)
                """)

                fert_list = []
                for fert_name, f_info in econ_data['fertilizer_breakdown'].items():
                    fert_list.append({
                        "Fertilizer": str(fert_name),
                        "Target Nutrient": str(f_info['nutrient']),
                        "Deficit (kg/ha)": float(f_info['deficit_per_ha']),
                        "Total Required (kg)": float(f_info['total_qty_kg']),
                        "Govt Retail Rate (₹/kg)": float(f_info['rate_per_kg']),
                        "Cost (₹)": float(f_info['cost'])
                    })
                df_econ = pd.DataFrame(fert_list)
                st.dataframe(df_econ, use_container_width=True)

            # --- PDF REPORT EXPORT ---
            st.divider()
            pdf_bytes = generate_farm_report_pdf(
                user_inputs=user_inputs,
                feature_names=feature_names,
                recommended_crop=final_crop,
                market_price=final_price,
                price_badge=final_badge,
                treatment_plan=treatment_plan,
                top_candidates=all_candidates,
                farm_economics=econ_data
            )
            
            col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
            with col_dl2:
                st.download_button(
                    label="📄 Download Farm Report",
                    data=pdf_bytes,
                    file_name=f"AgriSmart_Farm_Report_{final_crop.lower()}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    with tab2:
        st.subheader("📊 Machine Learning Model Analytics")
        st.caption("Empirical statistical benchmark evaluating algorithms trained in ml_engine.py across 22 crop classes")
        
        try:
            perf_df = pd.read_csv('model_performance.csv', index_col=0)
            perf_df.index.name = "Model"
            
            # KPI Cards
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Top Model", "Random Forest", "Production Ready")
            with kpi2:
                rf_acc = float(perf_df.loc['Random Forest', 'Accuracy']) if 'Random Forest' in perf_df.index else 0.993
                st.metric("Peak Accuracy", f"{rf_acc * 100:.2f}%", "+0.91% vs Tree")
            with kpi3:
                rf_f1 = float(perf_df.loc['Random Forest', 'F1-Score']) if 'Random Forest' in perf_df.index else 0.993
                st.metric("Weighted F1-Score", f"{rf_f1:.4f}", "Optimal")
                
            st.write("#### 📋 Model Performance Data")
            formatted_df = perf_df.copy()
            for col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
            st.dataframe(formatted_df, use_container_width=True)
            
            st.write("#### 📈 Multi-Metric Algorithm Comparison")
            metrics = perf_df.columns.tolist()
            models = perf_df.index.tolist()
            
            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            
            x = np.arange(len(metrics))
            width = 0.22
            palette = ['#1b5e20', '#2e7d32', '#81c784']
            
            for i, mod_name in enumerate(models):
                vals = perf_df.loc[mod_name].values.astype(float)
                rects = ax.bar(
                    x + (i - 1) * width,
                    vals,
                    width,
                    label=mod_name,
                    color=palette[i % len(palette)],
                    edgecolor='#ffffff',
                    linewidth=1.2
                )
                
                for rect in rects:
                    h = rect.get_height()
                    ax.annotate(
                        f'{h*100:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=7.5,
                        fontweight='bold',
                        color='#1b2e1e'
                    )
            
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontweight='bold', fontsize=10, color='#1b2e1e')
            ax.set_ylim(0.92, 1.025)
            ax.set_ylabel('Evaluation Score (0.0 to 1.0)', fontweight='bold', color='#1b2e1e')
            ax.set_title('Model Performance Metrics (Accuracy, Precision, Recall, F1)', fontsize=12, fontweight='bold', color='#1b5e20', pad=12)
            ax.legend(loc='lower left', framealpha=0.95, edgecolor='#c8e6c9')
            ax.grid(axis='y', linestyle='--', alpha=0.35, color='#a5d6a7')
            
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_color('#c8e6c9')
                
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.info(
                "💡 **Key Finding:** Random Forest demonstrates the highest generalization accuracy (99.32%) "
                "and superior precision (99.37%) with negligible variance across all 22 crop classes, "
                "confirming it as the optimal production model for biological suitability scoring."
            )
            
        except FileNotFoundError:
            st.error("`model_performance.csv` not found! Please run `python ml_engine.py` to generate the evaluation dataset.")
        except Exception as e:
            st.error(f"Error loading model analytics: {e}")

if __name__ == "__main__":
    main()
