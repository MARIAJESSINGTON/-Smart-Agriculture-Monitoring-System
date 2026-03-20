from flask import Flask, request, jsonify, render_template, render_template_string
from dotenv import load_dotenv
import sys, os, requests, time, folium

load_dotenv()

ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(ROOT_DIR, "src")
FRONTEND_DIR = os.path.join(ROOT_DIR, "Frontend")

sys.path.insert(0, SRC_DIR)
from predict import predict_crop, predict_irrigation

app = Flask(__name__, template_folder=FRONTEND_DIR)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL")
WEATHER_API_URL    = os.environ.get("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")
SOIL_API_URL       = os.environ.get("SOIL_API_URL",    "https://rest.isric.org/soilgrids/v2.0/properties/query")

FREE_MODELS = [
    "z-ai/glm-4.5-air:free",
]

# ─────────────────────────────────────────────
#  Core Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/crop", methods=["POST"])
def crop_prediction():
    try:
        data = request.get_json()
        features = [float(data["nitrogen"]), float(data["phosphorus"]),
                    float(data["potassium"]), float(data["temperature"]),
                    float(data["humidity"]),  float(data["ph"]),
                    float(data["rainfall"])]
        return jsonify({"success": True, "prediction": predict_crop(features)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/predict/irrigation", methods=["POST"])
def irrigation_prediction():
    try:
        data = request.get_json()
        features = [float(data["soil_moisture"]), float(data["soil_temp"]), float(data["humidity"])]
        return jsonify({"success": True, "prediction": predict_irrigation(features)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ─────────────────────────────────────────────
#  SIMPLE FOLIUM MAP ROUTE
# ─────────────────────────────────────────────

@app.route("/map", methods=["POST"])
def show_map():
    """
    Receives lat & lon, returns a simple Folium map HTML with a marker.
    """
    try:
        body = request.get_json()
        lat  = float(body["lat"])
        lon  = float(body["lon"])

        # Build simple Folium map
        m = folium.Map(location=[lat, lon], zoom_start=13)

        # Green marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"<b>📍 Your Location</b><br>Lat: {lat:.5f}<br>Lon: {lon:.5f}",
                max_width=200
            ),
            tooltip="📍 Farm Location",
            icon=folium.Icon(color="green", icon="leaf", prefix="fa")
        ).add_to(m)

        # Circle around point
        folium.Circle(
            location=[lat, lon],
            radius=400,
            color="#2d6a4f",
            fill=True,
            fill_color="#52b788",
            fill_opacity=0.2,
            weight=2
        ).add_to(m)

        return jsonify({"success": True, "map_html": m._repr_html_()})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ─────────────────────────────────────────────
#  GEO ANALYSIS HELPERS
# ─────────────────────────────────────────────

def fetch_soilgrids(lat, lon):
    params = {"lon": lon, "lat": lat,
              "property": ["clay","sand","silt","phh2o","soc","nitrogen","cec"],
              "depth": "0-5cm", "value": "mean"}
    resp = requests.get(SOIL_API_URL, params=params, timeout=12)
    resp.raise_for_status()
    raw  = resp.json()
    conv = {"clay":0.1,"sand":0.1,"silt":0.1,"phh2o":0.1,"soc":0.1,"nitrogen":0.01,"cec":0.1}
    results = {}
    for layer in raw.get("properties",{}).get("layers",[]):
        name = layer["name"]
        try:
            val = layer["depths"][0]["values"]["mean"]
            results[name] = round(val * conv.get(name,1), 2) if val is not None else None
        except: results[name] = None
    return results, "SoilGrids (ISRIC)"

def fetch_open_meteo_climate(lat, lon):
    params = {"latitude": lat, "longitude": lon,
              "daily": ["temperature_2m_max","precipitation_sum","relative_humidity_2m_max"],
              "timezone": "auto", "forecast_days": 7}
    resp = requests.get(WEATHER_API_URL, params=params, timeout=10)
    resp.raise_for_status()
    d = resp.json().get("daily", {})
    temps    = d.get("temperature_2m_max", [])
    precip   = d.get("precipitation_sum", [])
    humidity = d.get("relative_humidity_2m_max", [])
    return {
        "avg_temp":     round(sum(temps)/len(temps), 1)      if temps    else None,
        "total_precip": round(sum(precip), 1)                 if precip   else None,
        "avg_humidity": round(sum(humidity)/len(humidity), 1) if humidity else None,
    }

def estimate_soil_from_coordinates(lat, lon):
    abs_lat = abs(lat)
    if abs_lat <= 10:
        zone="tropical_humid"; clay,sand,silt=42,28,30; ph,soc,n,cec=5.4,12.0,0.8,14.0
    elif abs_lat <= 20:
        if 68<=lon<=90: zone="tropical_india"; clay,sand,silt=38,32,30; ph,soc,n,cec=6.8,8.5,0.6,22.0
        else:           zone="tropical_savanna"; clay,sand,silt=32,42,26; ph,soc,n,cec=6.0,9.0,0.5,12.0
    elif abs_lat <= 30:
        if 65<=lon<=90: zone="subtropical_india"; clay,sand,silt=28,38,34; ph,soc,n,cec=7.2,7.0,0.55,18.0
        else:           zone="subtropical"; clay,sand,silt=25,48,27; ph,soc,n,cec=7.0,6.0,0.4,14.0
    elif abs_lat <= 45:
        zone="temperate"; clay,sand,silt=22,40,38; ph,soc,n,cec=6.5,18.0,1.2,24.0
    else:
        zone="cool_temperate"; clay,sand,silt=18,35,47; ph,soc,n,cec=6.2,22.0,1.5,20.0
    return {"clay":clay,"sand":sand,"silt":silt,"phh2o":ph,"soc":soc,"nitrogen":n,"cec":cec,
            "_zone":zone,"_estimated":True}, f"Estimated (zone: {zone})"

def classify_soil_type(clay, sand, silt):
    if None in (clay,sand,silt): return "Unknown"
    if clay>=40: return "Clay"
    elif clay>=27 and sand<=20: return "Silty Clay"
    elif clay>=27 and sand<=45: return "Clay Loam"
    elif sand>=85 and silt<=15 and clay<=10: return "Sand"
    elif sand>=70 and clay<=15: return "Sandy Loam"
    elif silt>=80 and clay<=12: return "Silt"
    elif silt>=50 and clay<=27: return "Silt Loam"
    elif clay>=20 and clay<35 and sand>=45: return "Sandy Clay Loam"
    else: return "Loam"

def call_openrouter(prompt):
    if not OPENROUTER_API_KEY or "YOUR_NEW_KEY" in OPENROUTER_API_KEY:
        raise Exception("OpenRouter API key not set. Update your .env file.")
    last_error = None
    models = [OPENROUTER_MODEL] + [m for m in FREE_MODELS if m != OPENROUTER_MODEL]
    for model in models:
        try:
            print(f"[AgriSense] Trying: {model}")
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "http://localhost:5000",
                         "X-Title": "AgriSense"},
                json={"model": model,
                      "messages": [{"role":"user","content":prompt}],
                      "max_tokens": 1400},
                timeout=60
            )
            print(f"[AgriSense] {model} → {resp.status_code}")
            if resp.status_code == 401: raise Exception("Invalid API key. Get a new key from openrouter.ai/keys")
            if resp.status_code == 402: raise Exception("No credits. Add credits at openrouter.ai")
            if resp.status_code == 429: last_error=f"Rate limited: {model}"; time.sleep(2); continue
            if resp.status_code >= 400: last_error=f"{model} error {resp.status_code}"; continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"], model
        except requests.exceptions.Timeout:
            last_error=f"Timeout: {model}"; continue
        except Exception as e:
            if "API key" in str(e) or "credits" in str(e): raise
            last_error=str(e); continue
    raise Exception(f"All models failed. Last: {last_error}")

def ask_ai(soil_data, soil_type, lat, lon, data_source, climate=None):
    is_est = soil_data.get("_estimated", False)
    climate_str = ""
    if climate:
        climate_str = f"\nLive climate: Avg {climate.get('avg_temp')}°C, Rain {climate.get('total_precip')}mm/week, Humidity {climate.get('avg_humidity')}%\n"
    prompt = f"""You are an expert agronomist. Farmer land at lat:{lat}, lon:{lon}
{'Estimated soil (SoilGrids unavailable)' if is_est else 'Real soil data (SoilGrids ISRIC)'}:
Texture:{soil_type}, Clay:{soil_data.get('clay')}%, Sand:{soil_data.get('sand')}%, Silt:{soil_data.get('silt')}%
pH:{soil_data.get('phh2o')}, Organic Carbon:{soil_data.get('soc')}g/kg, Nitrogen:{soil_data.get('nitrogen')}g/kg, CEC:{soil_data.get('cec')}cmol/kg
{climate_str}
Provide:
1. **Soil Health Summary** — 3-4 sentences.
2. **Top 5 Recommended Crops** — one-line reason each.
3. **Best Crop Breeds & Varieties** — 2-3 cultivar names per crop.
4. **Soil Amendment Tips** — 3 actionable steps.
5. **Crops to Avoid** — 2-3 with brief reason.
Tailor to region at lat:{lat}, lon:{lon}."""
    return call_openrouter(prompt)

@app.route("/analyze/geo", methods=["POST"])
def geo_analysis():
    try:
        body = request.get_json()
        lat  = float(body["latitude"])
        lon  = float(body["longitude"])
        if not (-90<=lat<=90) or not (-180<=lon<=180):
            return jsonify({"success": False, "error": "Invalid coordinates."}), 400
        warnings = []
        try:
            soil_data, data_source = fetch_soilgrids(lat, lon)
        except Exception:
            warnings.append("SoilGrids unavailable. Using coordinate-based estimate.")
            soil_data, data_source = estimate_soil_from_coordinates(lat, lon)
        climate = None
        try: climate = fetch_open_meteo_climate(lat, lon)
        except: pass
        soil_type = classify_soil_type(soil_data.get("clay"), soil_data.get("sand"), soil_data.get("silt"))
        ai_report, model_used = ask_ai(soil_data, soil_type, lat, lon, data_source, climate)
        clean_soil = {k:v for k,v in soil_data.items() if not k.startswith("_")}
        return jsonify({"success":True, "coordinates":{"lat":lat,"lon":lon},
                        "soil_data":clean_soil, "soil_type":soil_type,
                        "data_source":data_source, "model_used":model_used,
                        "climate":climate, "ai_report":ai_report, "warnings":warnings})
    except requests.exceptions.Timeout:
        return jsonify({"success":False,"error":"Request timed out."}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"success":False,"error":f"API error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True) 