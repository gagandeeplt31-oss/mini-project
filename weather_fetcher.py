"""
weather_fetcher.py - Geolocation and Live Meteorological Data Fetcher
Provides IP-based location detection, OpenStreetMap/Nominatim geocoding,
and live environmental data retrieval via Open-Meteo API.
"""

import requests
import logging

logger = logging.getLogger(__name__)

IP_API_URL = "http://ip-api.com/json/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Polite User-Agent header for OpenStreetMap Nominatim compliance
GEOCODER_HEADERS = {
    "User-Agent": "AgriSmart-Crop-Decision-System/2.5 (contact: agrismart-support@farmtech.local)"
}


def detect_ip_location(timeout: float = 5.0) -> dict:
    """
    Detects the client's geographical location using the free ip-api.com service.
    
    Returns:
        dict with keys:
            'status': 'success' or 'error'
            'lat': float latitude or None
            'lon': float longitude or None
            'city': str or None
            'region': str or None
            'country': str or None
            'display': formatted location string
            'message': error message if failed
    """
    try:
        response = requests.get(IP_API_URL, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                city = data.get("city", "Unknown City")
                region = data.get("regionName", "")
                country = data.get("country", "")
                
                parts = [p for p in [city, region, country] if p]
                display_name = ", ".join(parts) if parts else "Current Location"
                
                return {
                    "status": "success",
                    "lat": float(data.get("lat")),
                    "lon": float(data.get("lon")),
                    "city": city,
                    "region": region,
                    "country": country,
                    "display": display_name,
                    "source": "IP Geolocation"
                }
            else:
                return {
                    "status": "error",
                    "lat": None,
                    "lon": None,
                    "message": data.get("message", "IP location lookup returned unhandled status.")
                }
    except requests.RequestException as e:
        logger.warning(f"IP Geolocation request failed: {e}")
        return {
            "status": "error",
            "lat": None,
            "lon": None,
            "message": f"Network error during IP detection: {e}"
        }
    except Exception as e:
        logger.error(f"Unexpected error in detect_ip_location: {e}")
        return {
            "status": "error",
            "lat": None,
            "lon": None,
            "message": f"Unexpected error: {e}"
        }

    return {
        "status": "error",
        "lat": None,
        "lon": None,
        "message": f"IP location lookup failed (HTTP {response.status_code})."
    }


def geocode_district(query: str, timeout: float = 5.0) -> dict:
    """
    Geocodes a user-entered city or district name into (latitude, longitude)
    using the free OpenStreetMap Nominatim search API.
    
    Returns:
        dict with keys:
            'status': 'success' or 'error'
            'lat': float latitude or None
            'lon': float longitude or None
            'display': clean formatted place name
            'message': error message if failed
    """
    clean_query = query.strip()
    if not clean_query:
        return {
            "status": "error",
            "lat": None,
            "lon": None,
            "message": "Query string is empty."
        }

    params = {
        "q": clean_query,
        "format": "json",
        "limit": 1,
        "addressdetails": 1
    }

    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=GEOCODER_HEADERS, timeout=timeout)
        if response.status_code == 200:
            results = response.json()
            if results and len(results) > 0:
                first = results[0]
                lat = float(first["lat"])
                lon = float(first["lon"])
                display = first.get("display_name", clean_query)
                
                # Truncate overly long display names
                parts = display.split(",")
                if len(parts) > 3:
                    short_display = ", ".join(p.strip() for p in parts[:3] + [parts[-1]])
                else:
                    short_display = display

                return {
                    "status": "success",
                    "lat": lat,
                    "lon": lon,
                    "display": short_display,
                    "source": "Nominatim Geocoder"
                }
            else:
                return {
                    "status": "error",
                    "lat": None,
                    "lon": None,
                    "message": f"Could not find coordinates for '{clean_query}'. Please verify spelling."
                }
    except requests.RequestException as e:
        logger.warning(f"Nominatim geocoding failed: {e}")
        return {
            "status": "error",
            "lat": None,
            "lon": None,
            "message": f"Geocoding network error: {e}"
        }
    except Exception as e:
        logger.error(f"Unexpected error in geocode_district: {e}")
        return {
            "status": "error",
            "lat": None,
            "lon": None,
            "message": f"Unexpected geocoding error: {e}"
        }

    return {
        "status": "error",
        "lat": None,
        "lon": None,
        "message": f"Geocoding service unavailable (HTTP {response.status_code})."
    }


def fetch_live_weather(lat: float, lon: float, timeout: float = 6.0) -> dict:
    """
    Calls the free Open-Meteo API to fetch current meteorological parameters:
    - temperature_2m (°C)
    - relative_humidity_2m (%)
    - precipitation (mm)
    
    Returns:
        dict with keys:
            'status': 'success' or 'error'
            'temperature': float
            'humidity': float
            'rainfall': float
            'units': dict of units
            'message': error message if failed
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "timezone": "auto"
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            units = data.get("current_units", {})
            
            temp = float(current.get("temperature_2m", 0.0))
            hum = float(current.get("relative_humidity_2m", 0.0))
            rain = float(current.get("precipitation", 0.0))
            
            return {
                "status": "success",
                "temperature": round(temp, 1),
                "humidity": round(hum, 1),
                "rainfall": round(rain, 1),
                "units": {
                    "temperature": units.get("temperature_2m", "°C"),
                    "humidity": units.get("relative_humidity_2m", "%"),
                    "rainfall": units.get("precipitation", "mm")
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Open-Meteo API returned HTTP {response.status_code}."
            }
    except requests.RequestException as e:
        logger.warning(f"Open-Meteo request failed: {e}")
        return {
            "status": "error",
            "message": f"Weather fetch network error: {e}"
        }
    except Exception as e:
        logger.error(f"Unexpected error in fetch_live_weather: {e}")
        return {
            "status": "error",
            "message": f"Unexpected weather error: {e}"
        }


if __name__ == "__main__":
    print("Testing IP Location...")
    ip_res = detect_ip_location()
    print("IP Result:", ip_res)
    
    if ip_res["status"] == "success":
        w = fetch_live_weather(ip_res["lat"], ip_res["lon"])
        print("Weather for IP:", w)

    print("\nTesting Geocoding for 'Nagpur'...")
    geo_res = geocode_district("Nagpur")
    print("Geo Result:", geo_res)
    if geo_res["status"] == "success":
        w2 = fetch_live_weather(geo_res["lat"], geo_res["lon"])
        print("Weather for Nagpur:", w2)
