"""
market_fetcher.py - Live Indian Mandi / Commodity Price Scraper with CSV Fallback.

Pulls live market prices for agricultural crops from public commodity tracking
portals (e.g., CommodityOnline). If an HTTP request fails, times out, or the
crop is not actively listed today, it seamlessly falls back to reading the
historical static prices in market_prices.csv.
"""

import re
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Badge Constants
BADGE_LIVE = "🟢 Live Mandi Rate"
BADGE_HISTORICAL = "🟡 Historical Data"

# Mapping of crop names to commodity tracking slugs / search terms
CROP_SLUGS = {
    'rice': ['rice'],
    'maize': ['maize'],
    'chickpea': ['bengal-gram-gram-whole', 'kabuli-chana-chickpeas-white', 'chana'],
    'kidneybeans': ['rajma', 'kidney-beans-rajma'],
    'pigeonpeas': ['arhar-turred-gram-whole', 'pegeon-pea-arhar-fali', 'tur'],
    'mothbeans': ['moth-dal', 'moth-beans'],
    'mungbean': ['green-gram-moong-whole', 'moong'],
    'blackgram': ['black-gram-urd-beans-whole', 'urad'],
    'lentil': ['lentil-masur-whole', 'masur', 'lentil'],
    'pomegranate': ['pomegranate'],
    'banana': ['banana'],
    'mango': ['mango'],
    'grapes': ['grapes'],
    'watermelon': ['water-melon', 'watermelon'],
    'muskmelon': ['karbuja-musk-melon', 'muskmelon'],
    'apple': ['apple'],
    'orange': ['orange'],
    'papaya': ['papaya'],
    'coconut': ['coconut'],
    'cotton': ['cotton'],
    'jute': ['jute'],
    'coffee': ['coffee']
}

# In-memory cache: {crop_lower: {'price': float, 'badge': str, 'is_live': bool, 'timestamp': float}}
_PRICE_CACHE = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes cache


def clean_price_string(val):
    """
    Cleans raw price text by stripping ₹ symbols, Rs, currency codes,
    commas, and whitespace. Converts quintal rates to per-kg rates when appropriate.
    Returns float or None.
    """
    if not val:
        return None
    val_str = str(val).strip()
    is_quintal = 'quintal' in val_str.lower()
    
    # Strip currency signs, labels and extract number
    clean_text = val_str.replace('₹', '').replace('Rs.', '').replace('Rs', '').replace('rs', '')
    match = re.search(r'[\d,]+(?:\.\d+)?', clean_text)
    if not match:
        return None
    
    raw_num = match.group(0).replace(',', '')
    try:
        price = float(raw_num)
        if is_quintal and price > 500:
            # 1 Quintal = 100 Kg; convert to per-kg rate
            price = price / 100.0
        elif price > 1000 and not is_quintal:
            # Mandi rate quoted in quintal without explicit unit
            price = price / 100.0
        return round(price, 2)
    except ValueError:
        return None


def fetch_live_crop_price(crop_name, timeout=3.5):
    """
    Scrapes today's live price for a crop from the public commodity tracking site.
    Returns float price if successful and positive, otherwise None.
    """
    crop_key = crop_name.lower().strip()
    slug_candidates = CROP_SLUGS.get(crop_key, [crop_key])
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    for slug in slug_candidates:
        url = f"https://www.commodityonline.com/mandiprices/{slug}"
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code != 200 or not response.text:
                continue
            
            # Guard against anti-bot / blocked pages
            soup = BeautifulSoup(response.text, 'html.parser')
            page_title = (soup.title.get_text() if soup.title else '').lower()
            if any(b in page_title for b in ['attention required', 'blocked', 'access denied', '403 forbidden', 'robot']):
                continue
            
            # Validate that the page is relevant to the target crop
            body_text = soup.get_text().lower()
            if crop_key not in body_text and slug not in body_text:
                continue
            
            # Pattern 1: Check "1 kg Price:" in tables
            for tr in soup.find_all('tr'):
                tr_text = tr.get_text()
                if '1 kg price' in tr_text.lower():
                    tds = tr.find_all('td')
                    if len(tds) >= 2:
                        parsed = clean_price_string(tds[1].get_text(strip=True))
                        if parsed and parsed > 0:
                            return parsed
                    parsed = clean_price_string(tr_text)
                    if parsed and parsed > 0:
                        return parsed
            
            # Pattern 2: Check mandi spot list items ("... is ₹X/Kg")
            for tr in soup.find_all('tr'):
                tr_text = tr.get_text()
                m = re.search(r'is\s*₹\s*([\d,]+(?:\.\d+)?)\s*/\s*kg', tr_text, re.IGNORECASE)
                if m:
                    parsed = clean_price_string(m.group(1))
                    if parsed and parsed > 0:
                        return parsed
            
            # Pattern 3: Avg Market Price / Quintal in summary table
            for tr in soup.find_all('tr'):
                tr_text = tr.get_text()
                if 'avg market price' in tr_text.lower():
                    parsed = clean_price_string(tr_text)
                    if parsed and parsed > 0:
                        return parsed
            
            # Pattern 4: Check specific commodity table rows
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 9:
                    commodity_cell = tds[0].get_text(strip=True).lower()
                    if crop_key in commodity_cell or any(s in commodity_cell for s in slug_candidates):
                        avg_text = tds[8].get_text(strip=True)
                        parsed = clean_price_string(avg_text)
                        if parsed and parsed > 0:
                            return parsed
                        
        except (requests.RequestException, Exception):
            # Try next slug candidate or fall through to fallback
            continue
            
    return None


def get_fallback_price(crop_name, fallback_csv="market_prices.csv"):
    """
    Reads the historical price for crop_name from market_prices.csv.
    """
    try:
        df = pd.read_csv(fallback_csv)
        df.columns = df.columns.str.strip()
        df.columns = ['Crop', 'Price'] + list(df.columns[2:])
        df['Crop'] = df['Crop'].astype(str).str.strip().str.lower()
        
        crop_clean = crop_name.lower().strip()
        match = df[df['Crop'] == crop_clean]
        if not match.empty:
            raw_price = match['Price'].values[0]
            cleaned = clean_price_string(str(raw_price))
            return float(cleaned) if cleaned is not None else float(raw_price)
    except Exception as e:
        print(f"[Warning] Fallback CSV read failed: {e}")
    return 0.0


def get_crop_price(crop_name, fallback_csv="market_prices.csv", force_refresh=False, timeout=3.5):
    """
    Main entry point for single crop price retrieval:
    Attempts to fetch live mandi price via web scraper.
    Seamlessly falls back to market_prices.csv if scraping fails, times out,
    or no valid price is listed today.
    
    Returns:
        tuple: (price: float, badge: str, is_live: bool)
    """
    crop_clean = crop_name.lower().strip()
    now = time.time()
    
    # 1. Check in-memory cache
    if not force_refresh and crop_clean in _PRICE_CACHE:
        entry = _PRICE_CACHE[crop_clean]
        if now - entry.get('timestamp', 0) < CACHE_TTL_SECONDS:
            return entry['price'], entry['badge'], entry['is_live']
    
    # 2. Attempt Live Scraping
    live_price = fetch_live_crop_price(crop_clean, timeout=timeout)
    if live_price is not None and live_price > 0:
        result = (round(live_price, 2), BADGE_LIVE, True)
    else:
        # 3. Robust Fallback to CSV
        fallback = get_fallback_price(crop_clean, fallback_csv=fallback_csv)
        result = (round(fallback, 2), BADGE_HISTORICAL, False)
        
    # Store in cache
    _PRICE_CACHE[crop_clean] = {
        'price': result[0],
        'badge': result[1],
        'is_live': result[2],
        'timestamp': now
    }
    return result


def get_all_market_prices(crops=None, fallback_csv="market_prices.csv", timeout=3.5):
    """
    Fetches market prices for multiple crops in parallel using ThreadPoolExecutor.
    Returns a DataFrame with columns: ['Crop', 'Price', 'Badge', 'Is_Live']
    """
    if crops is None:
        try:
            df_csv = pd.read_csv(fallback_csv)
            crops = df_csv.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        except Exception:
            crops = list(CROP_SLUGS.keys())
            
    results = []
    
    def _fetch_one(crop):
        price, badge, is_live = get_crop_price(crop, fallback_csv=fallback_csv, timeout=timeout)
        return {
            'Crop': crop.lower().strip(),
            'Price': price,
            'Badge': badge,
            'Is_Live': is_live
        }
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(_fetch_one, crops))
        
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("--- Testing market_fetcher.py ---")
    test_crops = ['rice', 'banana', 'apple', 'mothbeans', 'coffee']
    for c in test_crops:
        p, b, live = get_crop_price(c)
        print(f"Crop: {c:12} | Price: ₹{p:<6} | Status: {b} | Live: {live}")
