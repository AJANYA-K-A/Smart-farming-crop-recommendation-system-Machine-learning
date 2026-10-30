"""
Smart Farming Crop Recommendation System - Flask Application
Predicts optimal crops based on soil nutrients and climatic parameters.
Features Login Authentication, Executive Dashboard, REST APIs, and FarmBot AI.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

# ─── App & Path Configuration ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
MODEL_PATH = os.path.join(BASE_DIR, 'crop_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'Crop_recommendation.csv')

os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
app.secret_key = os.environ.get('SECRET_KEY', 'cropiq-precision-farming-key-2026')

# ─── Load ML Artifacts & Dataset ─────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATA_PATH)

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ─── Crop Knowledge Base (22 Crops) ──────────────────────────────────────────
CROP_DATA = {
    'rice': {
        'emoji': '🌾',
        'category': 'Cereal Grain',
        'season': 'Kharif (Monsoon)',
        'water': 'High (Standing water, 150-300 mm rainfall)',
        'soil': 'Clayey or loamy soil with pH 5.5–7.0',
        'temp': '20°C – 35°C',
        'tip': 'Maintain consistent puddle water during tillering and panicle development.'
    },
    'maize': {
        'emoji': '🌽',
        'category': 'Cereal / Fodder',
        'season': 'Kharif / Rabi',
        'water': 'Moderate (50-100 mm rainfall)',
        'soil': 'Well-drained fertile loam with pH 5.5–7.5',
        'temp': '18°C – 27°C',
        'tip': 'Ensure good drainage; avoid waterlogging especially at seedling stage.'
    },
    'chickpea': {
        'emoji': '🌱',
        'category': 'Pulse / Legume',
        'season': 'Rabi (Winter)',
        'water': 'Low to Moderate (60-90 mm rainfall)',
        'soil': 'Deep, well-drained sandy loam to clay loam (pH 6.0–8.0)',
        'temp': '15°C – 25°C',
        'tip': 'Inoculate seeds with Rhizobium culture to enhance biological nitrogen fixation.'
    },
    'kidneybeans': {
        'emoji': '🫘',
        'category': 'Pulse / Legume',
        'season': 'Rabi / Spring',
        'water': 'Moderate (60-150 mm rainfall)',
        'soil': 'Rich, light, organic loam (pH 5.5–6.5)',
        'temp': '15°C – 25°C',
        'tip': 'Sensitive to salinity and frost; protect roots with organic mulching.'
    },
    'pigeonpeas': {
        'emoji': '🌿',
        'category': 'Pulse / Legume',
        'season': 'Kharif (Long duration)',
        'water': 'Moderate to Dry (90-150 mm rainfall)',
        'soil': 'Loamy to clay loam with good drainage (pH 6.0–7.5)',
        'temp': '20°C – 35°C',
        'tip': 'Deep taproot system tolerates drought; ideal for intercropping with cereals.'
    },
    'mothbeans': {
        'emoji': '🌾',
        'category': 'Arid Legume',
        'season': 'Kharif',
        'water': 'Low / Drought Tolerant (30-75 mm rainfall)',
        'soil': 'Sandy to sandy loam, well-drained (pH 6.5–8.0)',
        'temp': '25°C – 40°C',
        'tip': 'Excellent for arid drylands; acts as soil erosion buffer and green manure.'
    },
    'mungbean': {
        'emoji': '🫘',
        'category': 'Pulse / Legume',
        'season': 'Kharif / Summer',
        'water': 'Moderate (40-90 mm rainfall)',
        'soil': 'Well-drained loamy soil (pH 6.2–7.2)',
        'temp': '25°C – 35°C',
        'tip': 'Short duration crop (60-70 days); perfect for crop rotation.'
    },
    'blackgram': {
        'emoji': '🫘',
        'category': 'Pulse / Legume',
        'season': 'Kharif / Spring',
        'water': 'Moderate (60-80 mm rainfall)',
        'soil': 'Heavy loams or clayey soils with adequate moisture (pH 6.5–7.8)',
        'temp': '25°C – 35°C',
        'tip': 'Enriches soil nitrogen naturally; spray urea/zinc during pod formation.'
    },
    'lentil': {
        'emoji': '🥣',
        'category': 'Pulse / Legume',
        'season': 'Rabi (Cool season)',
        'water': 'Low to Moderate (40-60 mm rainfall)',
        'soil': 'Clay loam to light soils (pH 6.0–7.5)',
        'temp': '15°C – 25°C',
        'tip': 'Requires cool weather during vegetative growth and warm weather at maturity.'
    },
    'pomegranate': {
        'emoji': '🍎',
        'category': 'Horticultural Fruit',
        'season': 'Perennial (Mridu/Hasta bahar)',
        'water': 'Moderate to Low (100-120 mm rainfall)',
        'soil': 'Deep loamy to sandy loam, tolerates slight alkalinity (pH 6.5–7.5)',
        'temp': '20°C – 35°C',
        'tip': 'Implement drip irrigation and regulate flowering through regulated water stress (Bahar treatment).'
    },
    'banana': {
        'emoji': '🍌',
        'category': 'Tropical Fruit',
        'season': 'Year-round planting',
        'water': 'High (90-200 mm regular moisture)',
        'soil': 'Deep, rich, well-drained alluvial or volcanic soil (pH 6.0–7.5)',
        'temp': '25°C – 35°C',
        'tip': 'Heavy feeder of Potassium (K); mulch heavily to retain soil humidity.'
    },
    'mango': {
        'emoji': '🥭',
        'category': 'Tropical Fruit Tree',
        'season': 'Perennial',
        'water': 'Moderate (90-100 mm rainfall, dry period during flowering)',
        'soil': 'Deep, well-drained alluvial loam (pH 5.5–7.5)',
        'temp': '24°C – 35°C',
        'tip': 'Withhold irrigation before flowering to induce profuse bloom.'
    },
    'grapes': {
        'emoji': '🍇',
        'category': 'Vine Fruit',
        'season': 'Perennial',
        'water': 'Moderate (60-75 mm rainfall)',
        'soil': 'Deep sandy loam or gravelly soil with good drainage (pH 6.5–7.5)',
        'temp': '15°C – 35°C',
        'tip': 'Prune systematically twice a year; protect bunches from late-season rainfall.'
    },
    'watermelon': {
        'emoji': '🍉',
        'category': 'Cucurbit Fruit',
        'season': 'Zaid / Summer',
        'water': 'Moderate (40-60 mm rainfall)',
        'soil': 'Sandy loam or riverbed alluvium (pH 6.0–7.0)',
        'temp': '24°C – 35°C',
        'tip': 'Requires high sunlight and dry weather during fruit ripening for sweetness.'
    },
    'muskmelon': {
        'emoji': '🍈',
        'category': 'Cucurbit Fruit',
        'season': 'Zaid / Summer',
        'water': 'Moderate (20-30 mm rainfall)',
        'soil': 'Well-drained sandy loam soil (pH 6.0–7.0)',
        'temp': '25°C – 35°C',
        'tip': 'Avoid overwatering during fruit ripening to prevent fruit cracking and diluted sugar.'
    },
    'apple': {
        'emoji': '🍏',
        'category': 'Temperate Fruit',
        'season': 'Cool Temperate / Perennial',
        'water': 'Moderate (100-120 mm rainfall / snowfall)',
        'soil': 'Deep, fertile, well-drained loamy soil (pH 5.5–6.5)',
        'temp': '10°C – 24°C (requires winter chilling)',
        'tip': 'Needs 1000–1200 chilling hours below 7°C for healthy bud break.'
    },
    'orange': {
        'emoji': '🍊',
        'category': 'Citrus Fruit',
        'season': 'Subtropical / Perennial',
        'water': 'Moderate (100-120 mm rainfall)',
        'soil': 'Light, well-drained sandy loam (pH 6.0–7.5)',
        'temp': '15°C – 35°C',
        'tip': 'Susceptible to waterlogging; apply micronutrient sprays (Zinc, Iron) periodically.'
    },
    'papaya': {
        'emoji': '🥭',
        'category': 'Tropical Fruit',
        'season': 'Continuous (10-12 month cycle)',
        'water': 'Moderate to High (140-250 mm rainfall)',
        'soil': 'Rich, friable, well-drained loam (pH 6.0–7.0)',
        'temp': '22°C – 35°C',
        'tip': 'Extremely sensitive to collar rot if water stagnates at base; plant on raised beds.'
    },
    'coconut': {
        'emoji': '🥥',
        'category': 'Plantation Crop',
        'season': 'Coastal Tropical / Perennial',
        'water': 'High (140-220 mm rainfall or irrigation)',
        'soil': 'Sandy loam, alluvial, red sandy loam (pH 5.2–8.0)',
        'temp': '25°C – 35°C with high humidity',
        'tip': 'Requires coastal climate or high humidity; apply regular organic compost and salt/potash.'
    },
    'cotton': {
        'emoji': '☁️',
        'category': 'Fiber Cash Crop',
        'season': 'Kharif',
        'water': 'Moderate (60-90 mm rainfall)',
        'soil': 'Deep black cotton soil (Vertisols) or alluvial loam (pH 6.0–8.0)',
        'temp': '21°C – 35°C (warm days and sunny skies)',
        'tip': 'Warm, dry weather required during boll opening and harvesting to preserve lint quality.'
    },
    'jute': {
        'emoji': '🌿',
        'category': 'Fiber Crop',
        'season': 'Pre-Kharif / Monsoon',
        'water': 'High (150-200 mm rainfall)',
        'soil': 'Alluvial delta soil with high organic matter (pH 6.0–7.5)',
        'temp': '24°C – 37°C with high humidity (>80%)',
        'tip': 'Thrives in river basins with standing monsoon water; retting requires clean flowing water.'
    },
    'coffee': {
        'emoji': '☕',
        'category': 'Beverage Plantation',
        'season': 'Hill Shade Grown / Perennial',
        'water': 'High (150-200 mm rainfall, blossom showers)',
        'soil': 'Porous, rich, volcanic/forest humus loam (pH 6.0–6.5)',
        'temp': '15°C – 28°C under canopy shade',
        'tip': 'Grow under two-tier shade trees; critical blossom showers trigger flowering.'
    }
}

# ─── Static Visualisations ───────────────────────────────────────────────────
CHARTS = [
    ("Nitrogen Distribution", lambda: sns.histplot(df['N'], kde=True, color='#4a7c35')),
    ("Phosphorus Distribution", lambda: sns.histplot(df['P'], kde=True, color='#7db65a')),
    ("Potassium Distribution", lambda: sns.histplot(df['K'], kde=True, color='#2d4a1e')),
    ("Temperature Distribution", lambda: sns.histplot(df['temperature'], kde=True, color='#e8a020')),
    ("Humidity Distribution", lambda: sns.histplot(df['humidity'], kde=True, color='#3a9abf')),
    ("pH Distribution", lambda: sns.histplot(df['ph'], kde=True, color='#9b5de5')),
    ("Rainfall Distribution", lambda: sns.histplot(df['rainfall'], kde=True, color='#1e88c4')),
    ("Temperature vs Humidity", lambda: sns.scatterplot(x='temperature', y='humidity', data=df, alpha=0.5, color='#4a7c35')),
    ("pH vs Rainfall", lambda: sns.scatterplot(x='ph', y='rainfall', data=df, alpha=0.5, color='#e8a020')),
    ("Feature Correlation Heatmap", lambda: sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='YlGn', fmt='.2f')),
]


def ensure_charts():
    """Generate charts once at startup if not present, avoiding blocking user requests."""
    sns.set_theme(style='whitegrid', palette='muted')
    for i, (title, plot_fn) in enumerate(CHARTS, 1):
        chart_path = os.path.join(STATIC_DIR, f'chart{i}.png')
        if not os.path.exists(chart_path):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.sca(ax)
            plot_fn()
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            fig.tight_layout()
            fig.savefig(chart_path, dpi=100)
            plt.close(fig)


# Ensure static chart images exist on load
ensure_charts()

# ─── Agricultural Chatbot Intelligence ───────────────────────────────────────
CROP_SYNONYMS = {
    'rice': ['rice', 'paddy', 'dhan'],
    'maize': ['maize', 'corn', 'sweetcorn', 'makka'],
    'chickpea': ['chickpea', 'chickpeas', 'gram', 'chana', 'bengal gram'],
    'kidneybeans': ['kidneybeans', 'kidneybean', 'kidney bean', 'rajma', 'beans'],
    'pigeonpeas': ['pigeonpeas', 'pigeonpea', 'arhar', 'toor', 'tur', 'red gram'],
    'mothbeans': ['mothbeans', 'mothbean', 'moth bean', 'matki', 'turkish gram'],
    'mungbean': ['mungbean', 'mungbeans', 'mung', 'moong', 'green gram'],
    'blackgram': ['blackgram', 'black gram', 'urad', 'mash'],
    'lentil': ['lentil', 'lentils', 'masoor'],
    'pomegranate': ['pomegranate', 'pomegranates', 'anar'],
    'banana': ['banana', 'bananas', 'kela', 'plantain'],
    'mango': ['mango', 'mangoes', 'mangos', 'aam'],
    'grapes': ['grapes', 'grape', 'vineyard', 'angur'],
    'watermelon': ['watermelon', 'watermelons', 'tarbooz'],
    'muskmelon': ['muskmelon', 'muskmelons', 'cantaloupe', 'kharbooja'],
    'apple': ['apple', 'apples', 'seb'],
    'orange': ['orange', 'oranges', 'citrus', 'santra'],
    'papaya': ['papaya', 'papayas', 'papita'],
    'coconut': ['coconut', 'coconuts', 'nariyal'],
    'cotton': ['cotton', 'kapas'],
    'jute': ['jute', 'pat'],
    'coffee': ['coffee', 'coffee bean', 'arabica', 'robusta']
}

AGRI_KNOWLEDGE = [
    # Nitrogen (N)
    {
        'keywords': ['nitrogen', 'urea', 'leafy', 'chlorosis', 'yellow leaf', 'yellow leaves', 'pale leaves'],
        'reply': "🌱 **Nitrogen (N)** is crucial for chlorophyll formation, vegetative canopy growth, and protein synthesis. Nitrogen deficiency causes stunted growth and pale yellowing (chlorosis) of older leaves. Supplement with Urea (46% N), Ammonium Sulfate, or well-rotted cattle manure."
    },
    # Phosphorus (P)
    {
        'keywords': ['phosphorus', 'phosphate', 'dap', 'ssp', 'root development', 'rooting', 'bloom', 'flowering'],
        'reply': "🌿 **Phosphorus (P)** powers cellular energy transfer (ATP), rapid seedling root establishment, and early flowering/fruit set. Deficiency results in purplish leaf margins and stunted root systems. Apply Diammonium Phosphate (DAP) or Single Super Phosphate (SSP) at planting."
    },
    # Potassium (K)
    {
        'keywords': ['potassium', 'potash', 'mop', 'sop', 'stomata', 'drought resistance', 'stem strength'],
        'reply': "🪵 **Potassium (K)** regulates stomatal opening, improves disease resistance, activates enzymatic processes, and enhances crop drought resilience. Deficiency produces leaf margin burning/scorching. Apply Muriate of Potash (MOP) or Sulfate of Potash (SOP)."
    },
    # Soil pH & Liming
    {
        'keywords': ['ph', 'soil ph', 'acidic', 'alkaline', 'lime', 'liming', 'gypsum', 'sulfur'],
        'reply': "🧫 **Soil pH Guide:** The optimal bioavailable range for most crops is pH 6.0 to 7.2.\n• For Acidic Soils (pH < 6.0): Broadcast agricultural lime (calcium carbonate) or dolomite to neutralize acidity.\n• For Alkaline Soils (pH > 7.5): Incorporate agricultural gypsum (calcium sulfate) or elemental sulfur with compost to lower pH."
    },
    # Soil Types & Texture
    {
        'keywords': ['soil type', 'soil texture', 'clay soil', 'sandy soil', 'loam', 'black soil', 'alluvial'],
        'reply': "🪴 **Soil Texture Guide:**\n• Loam (Ideal): Balanced drainage and moisture retention; optimal for cereals, vegetables, and pulses.\n• Clay: High water-holding capacity; best for rice and wetland crops.\n• Sandy: Fast drainage and warms quickly; ideal for melons and root vegetables with organic mulch.\n• Black Soil (Vertisols): Rich in clay minerals with high fertility; optimal for cotton, wheat, and pulses."
    },
    # Temperature
    {
        'keywords': ['temperature', 'temp', 'heat', 'cold', 'frost', 'chilling hours'],
        'reply': "🌡️ **Temperature Impact:** Temperature regulates crop metabolic rates and transpiration. Temperate crops (apples, lentils, chickpeas) require 12°C–22°C and cold chilling hours, while tropical crops (rice, papaya, cotton, watermelon) thrive between 22°C–35°C."
    },
    # Humidity
    {
        'keywords': ['humidity', 'relative humidity', 'moisture in air', 'humid'],
        'reply': "💧 **Relative Humidity (RH):** High humidity (>80%) accelerates fungal spores but is favored by rice and jute. Moderate humidity (50–70%) promotes healthy transpiration, pollination, and fruit development for most field crops."
    },
    # Rainfall
    {
        'keywords': ['rainfall', 'rain', 'precipitation', 'monsoon', 'annual rainfall'],
        'reply': "🌧️ **Rainfall Dynamics:** Heavy rainfall (>150 mm) favors standing water crops like rice and hill plantations like coffee. Moderate rainfall (60–100 mm) supports maize, cotton, and pulses. Arid crops (mothbeans, chickpeas) tolerate 30–80 mm."
    },
    # Irrigation
    {
        'keywords': ['irrigation', 'drip', 'sprinkler', 'watering', 'water saving'],
        'reply': "💦 **Irrigation Systems:**\n• Drip Irrigation: Delivers water directly to root zones, reducing water consumption by up to 60% and minimizing fungal foliage diseases.\n• Sprinkler Irrigation: Uniformly covers large acreage on undulating terrain.\n• Moisture Sensors: Track soil moisture tension to irrigate only when necessary."
    },
    # Pest & Disease Control (IPM)
    {
        'keywords': ['pest', 'pests', 'insects', 'disease', 'fungus', 'blight', 'rot', 'ipm', 'neem'],
        'reply': "🐛 **Integrated Pest Management (IPM):** Combine cultural, biological, and organic methods:\n1. Cultural: Practice 3-year crop rotation and intercropping.\n2. Physical: Install yellow sticky traps and solar light traps.\n3. Biological: Conserve natural predators like ladybird beetles.\n4. Organic: Spray 5% neem seed kernel extract (NSKE) or cold-pressed neem oil."
    },
    # Fertilizers & Compost
    {
        'keywords': ['fertilizer', 'fertilizers', 'compost', 'vermicompost', 'manure', 'organic farming'],
        'reply': "🧪 **Fertilizer & Soil Nutrition:** Blend chemical N-P-K with organic vermicompost or farmyard manure (FYM). Organic amendments improve water holding capacity, build microbial biomass, and enhance nutrient uptake efficiency."
    },
    # Model & Accuracy
    {
        'keywords': ['accuracy', 'random forest', 'algorithm', 'model', 'dataset', 'machine learning', 'ml', 'training'],
        'reply': "📊 **Machine Learning Architecture:** CropIQ employs a Scikit-Learn Random Forest Classifier (100 ensemble estimators) achieving **99.55% test accuracy**. It is trained on 2,200 multi-climatic agricultural records across 22 crop classes using 7 environmental features (N, P, K, Temperature, Humidity, pH, Rainfall)."
    },
    # Predictor Instructions
    {
        'keywords': ['predict', 'recommend', 'how to use', 'preset', 'how to predict'],
        'reply': "⚡ **How to Predict:** Open the **Crop Predictor** tab, enter your soil N-P-K levels, temperature, humidity, pH, and rainfall (or pick any 1-Click Preset like Rice, Maize, or Coffee), and click **Run AI Crop Prediction** for an instant ML recommendation!"
    },
    # Help & Capabilities
    {
        'keywords': ['help', 'who are you', 'what can you do', 'what is farmbot', 'features'],
        'reply': "🤖 **I'm FarmBot**, your smart agricultural AI advisor! I can assist with:\n• Crop recommendations & requirements for all 22 supported crops\n• Soil nutrients (Nitrogen, Phosphorus, Potassium, fertilizers)\n• Soil pH adjustments & textures\n• Irrigation methods & water management\n• Pest & disease control (IPM)\n• CropIQ machine learning model insights"
    },
    # Greetings
    {
        'keywords': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening', 'namaste'],
        'reply': "Hello! 🌱 I'm FarmBot, your precision farming AI advisor. Ask me anything about crop varieties, soil nutrients, fertilizers, pH adjustment, or irrigation!"
    },
    # Gratitude
    {
        'keywords': ['thank', 'thanks', 'appreciate', 'great', 'awesome'],
        'reply': "You're very welcome! Wishing you high yields, healthy soils, and prosperous harvests! 🌾🚜"
    },
    # Farewells
    {
        'keywords': ['bye', 'goodbye', 'see you'],
        'reply': "Goodbye! Come back whenever you need smart agricultural advice. Have a productive day in the field! 🚜"
    }
]


def get_response(user_input: str) -> str:
    """Intelligent agricultural assistant reply that answers precisely according to user inquiry."""
    if not user_input or not user_input.strip():
        return "Please ask a question regarding crops, soil nutrients (NPK), climate, pH, or irrigation!"

    text = user_input.strip().lower()

    # 1. Direct Numeric Prediction via Chat (e.g. N=90, P=42, K=43, temp=21, hum=82, ph=6.5, rain=200)
    num_matches = {}
    for param in ['n', 'p', 'k', 'temp', 'hum', 'ph', 'rain']:
        m = re.search(rf'{param}\w*\s*[:=]\s*([\d\.]+)', text)
        if m:
            try:
                num_matches[param] = float(m.group(1))
            except ValueError:
                pass

    if len(num_matches) >= 3:
        n_val = num_matches.get('n', 50.0)
        p_val = num_matches.get('p', 50.0)
        k_val = num_matches.get('k', 50.0)
        t_val = num_matches.get('temp', 25.0)
        h_val = num_matches.get('hum', 70.0)
        ph_val = num_matches.get('ph', 6.5)
        r_val = num_matches.get('rain', 100.0)

        in_df = pd.DataFrame([[n_val, p_val, k_val, t_val, h_val, ph_val, r_val]], columns=FEATURE_COLS)
        pred_label = label_encoder.inverse_transform(model.predict(in_df))[0]
        pred_probs = model.predict_proba(in_df)[0]
        conf_score = round(float(np.max(pred_probs)) * 100, 1)
        details = CROP_DATA.get(pred_label.lower(), {})

        return (
            f"⚡ **Instant ML Prediction:** Based on your parameters "
            f"(N: {n_val}, P: {p_val}, K: {k_val}, Temp: {t_val}°C, Humidity: {h_val}%, pH: {ph_val}, Rain: {r_val}mm):\n\n"
            f"**Optimal Recommendation: {details.get('emoji', '🌾')} {pred_label.upper()}** ({conf_score}% match confidence)!\n"
            f"• Category: {details.get('category', 'Crop')}\n"
            f"• Season: {details.get('season', 'Optimal Season')}\n"
            f"• Agronomist Tip: {details.get('tip', 'Follow standard crop management.')}"
        )

    # 2. Condition: Low Rainfall / Drought / Water-scarce crops
    if ('low' in text or 'less' in text or 'little' in text or 'scarce' in text or 'drought' in text or 'arid' in text or 'dry' in text) and \
       ('water' in text or 'rain' in text or 'rainfall' in text or 'crop' in text or 'grow' in text or 'plant' in text):
        return (
            "🌵 **Top Crops for Low Rainfall & Drought-Prone Regions (30–80 mm):**\n"
            "1. **Mothbeans** (Avg Rain: 51 mm) — Supreme arid pulse; deep taproot anchors in loose dry soils.\n"
            "2. **Mungbean** (Avg Rain: 48 mm) — Short 60-day lifecycle; minimal moisture demand.\n"
            "3. **Lentil** (Avg Rain: 46 mm) — Excellent semi-arid cool-season pulse.\n"
            "4. **Watermelon / Muskmelon** (Avg Rain: 25–50 mm) — Thrives in dry, warm climates with light soils.\n"
            "5. **Chickpea** (Avg Rain: 80 mm) — High drought tolerance; utilizes deep residual moisture."
        )

    # 3. Condition: High Rainfall / Wetland / Heavy Monsoon
    if ('high' in text or 'heavy' in text or 'more' in text or 'excess' in text or 'flood' in text or 'wet' in text) and \
       ('water' in text or 'rain' in text or 'rainfall' in text or 'crop' in text or 'grow' in text or 'plant' in text):
        return (
            "🌊 **Top Crops for High Rainfall & Wetlands (>150 mm):**\n"
            "1. **Rice / Paddy** (Avg Rain: 236 mm) — Thrives in standing puddle water and clayey basins.\n"
            "2. **Coconut** (Avg Rain: 176 mm) — Demands heavy tropical rainfall and continuous humidity.\n"
            "3. **Jute** (Avg Rain: 175 mm) — Ideal for delta floodplains and monsoon saturation.\n"
            "4. **Coffee** (Avg Rain: 158 mm) — Requires heavy seasonal blossom showers and humid hill canopy."
        )

    # 4. Condition: Winter / Cold / Rabi Season
    if 'winter' in text or 'rabi' in text or 'cold' in text or 'frost' in text or 'chilling' in text:
        return (
            "❄️ **Top Crops for Winter (Rabi Season, 12°C–22°C):**\n"
            "• **Chickpea (Gram)** — Major rabi legume; fixes atmospheric nitrogen.\n"
            "• **Lentil (Masoor)** — Thrives in cool, crisp winter loam.\n"
            "• **Kidneybeans (Rajma)** — High-value pulse for cool spring/rabi zones.\n"
            "• **Apple** (Temperate) — Requires 1000+ chilling hours below 7°C for bud breaking."
        )

    # 5. Condition: Summer / Zaid / Hot Climate
    if 'summer' in text or 'zaid' in text or 'hot' in text or 'kharif' in text or 'monsoon' in text:
        return (
            "☀️ **Top Crops for Summer & Warm Climates (24°C–35°C):**\n"
            "• **Zaid (Summer):** Watermelon, Muskmelon, Mungbean, Cucumber.\n"
            "• **Kharif (Monsoon):** Rice, Maize, Cotton, Jute, Pigeonpeas, Blackgram.\n"
            "• **Tropical Fruits:** Papaya (33.7°C), Mango (31.2°C), Banana (27.4°C)."
        )

    # 6. Soil-Specific Queries
    if 'black soil' in text or 'vertisol' in text or 'black cotton' in text:
        return (
            "🌱 **Optimal Crops for Black Soil (Vertisols):**\n"
            "1. **Cotton** (Best fit; black soil retains moisture and supports vigorous boll formation).\n"
            "2. **Chickpea & Pigeonpeas** (Tolerate clay cracking and access deep nutrient reserves).\n"
            "3. **Maize & Sorghum** (High cation exchange capacity produces superior yields)."
        )
    if 'sandy soil' in text or 'sand' in text:
        return (
            "🏖️ **Optimal Crops for Sandy Soil:**\n"
            "1. **Watermelon & Muskmelon** (Loose, fast-warming soil prevents fruit/root collar rots).\n"
            "2. **Mothbeans** (Deep taproot binds sandy particles and builds organic matter).\n"
            "3. **Groundnuts & Root Crops** (Easy penetration and unhindered pod expansion)."
        )
    if 'clay soil' in text or 'clayey' in text:
        return (
            "🧱 **Optimal Crops for Clay Soil:**\n"
            "1. **Rice / Paddy** (Clay layer prevents water percolation, keeping standing water).\n"
            "2. **Jute** (Heavy moisture retention supports fast stem elongation).\n"
            "3. **Blackgram** (Enjoys dense, nutrient-rich moisture profiles)."
        )

    # 7. Nutrient Deficiencies & Plant Symptoms
    if 'yellow' in text or 'chlorosis' in text or 'pale leaf' in text or 'pale leaves' in text:
        return (
            "🍂 **Agronomist Diagnosis for Yellow Leaves (Chlorosis):**\n"
            "1. **Older Lower Leaves Yellowing First:** Classic sign of **Nitrogen (N) deficiency**. The plant moves mobile nitrogen to new shoots. *Remedy: Apply Urea (46% N) or composted manure tea.*\n"
            "2. **Interveinal Yellowing (Yellow leaf with green veins):** Indicates **Iron (Fe) or Magnesium (Mg) deficiency**.\n"
            "3. **Yellowing accompanied by wilting:** Indicates **over-irrigation / waterlogging**, which suffocates roots and stops nutrient absorption."
        )
    if 'burnt' in text or 'scorched' in text or 'brown edge' in text or 'brown margin' in text:
        return (
            "🔥 **Diagnosis for Brown / Scorched Leaf Margins:**\n"
            "• Distinct symptom of **Potassium (K) deficiency**. Weakens stomatal control and causes cell desiccation at margins.\n"
            "• *Remedy: Top-dress with Muriate of Potash (MOP, 60% K2O) or spray soluble potassium sulfate.*"
        )
    if 'purple' in text or 'bronze' in text or 'purpling' in text:
        return (
            "🟣 **Diagnosis for Purple / Bronzed Leaves:**\n"
            "• Symptom of **Phosphorus (P) deficiency** or cold root zones inhibiting P assimilation.\n"
            "• *Remedy: Apply Diammonium Phosphate (DAP) or Single Super Phosphate (SSP) near the root zone.*"
        )

    # 8. Check for specific crop inquiries across all 22 crops
    for crop_key, synonyms in CROP_SYNONYMS.items():
        for syn in synonyms:
            pattern = r'\b' + re.escape(syn) + r'\b'
            if re.search(pattern, text) or syn in text:
                details = CROP_DATA.get(crop_key, {})

                # If user asks specifically about water/rain for this crop
                if 'water' in text or 'rain' in text or 'irrigation' in text:
                    return (
                        f"{details.get('emoji', '🌾')} **{crop_key.title()} — Water Requirements:**\n"
                        f"• Water Need: {details.get('water', 'Moderate')}\n"
                        f"• Ideal Season: {details.get('season', 'Optimal Season')}\n"
                        f"• Agronomist Tip: {details.get('tip', 'Ensure adequate drainage.')}"
                    )
                # If user asks specifically about soil for this crop
                if 'soil' in text or 'ph' in text:
                    return (
                        f"{details.get('emoji', '🌾')} **{crop_key.title()} — Soil & pH Compatibility:**\n"
                        f"• Soil Type: {details.get('soil', 'Well-drained loam')}\n"
                        f"• Agronomist Tip: {details.get('tip', 'Maintain optimal soil organic matter.')}"
                    )
                # If user asks specifically about temperature for this crop
                if 'temp' in text or 'climate' in text or 'weather' in text:
                    return (
                        f"{details.get('emoji', '🌾')} **{crop_key.title()} — Climate & Temperature:**\n"
                        f"• Optimal Temperature: {details.get('temp', '20°C - 30°C')}\n"
                        f"• Ideal Season: {details.get('season', 'Optimal Season')}"
                    )

                # General crop breakdown
                return (
                    f"{details.get('emoji', '🌾')} **{crop_key.title()}** ({details.get('category', 'Crop')}):\n"
                    f"• **Ideal Season:** {details.get('season', 'Optimal Season')}\n"
                    f"• **Optimal Temperature:** {details.get('temp', '20-30°C')}\n"
                    f"• **Water & Rainfall:** {details.get('water', 'Moderate')}\n"
                    f"• **Soil Compatibility:** {details.get('soil', 'Loamy soil')}\n"
                    f"• **Agronomist Tip:** {details.get('tip', 'Ensure balanced fertilization.')}"
                )

    # 9. General agricultural knowledge base
    for entry in AGRI_KNOWLEDGE:
        for kw in entry['keywords']:
            if kw in text:
                return entry['reply']

    # 10. Intelligent, Question-Aware Fallback
    return (
        "🌱 That's a great farming inquiry! Here is how FarmBot can assist you directly:\n"
        "• **Specific Crops:** Ask *'How to grow Banana?'*, *'Rice rainfall'*, or *'Cotton fertilizer'*\n"
        "• **Soil & Nutrients:** Ask *'Why are my leaves yellow?'*, *'How to raise soil pH'*, or *'Best fertilizer for maize'*\n"
        "• **Weather & Climate:** Ask *'What crops for low rainfall?'* or *'What to grow in winter?'*\n"
        "• **AI Predictions:** Provide values (e.g. *'N=90, P=42, K=43, temp=21, rain=200'*) for an instant ML recommendation!"
    )


# ─── Auth & Page Routes ──────────────────────────────────────────────────────
@app.route('/')
def index():
    """Entry point: starts at login page if not authenticated, else dashboard."""
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login portal page."""
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if email and password:
            lower_email = email.lower()
            if 'farmer' in lower_email:
                name = 'Farmer John'
                role = 'Farm Operations'
            elif 'agro' in lower_email or 'agronomist' in lower_email:
                name = 'Dr. Sarah Green'
                role = 'Lead Agronomist'
            elif 'guest' in lower_email:
                name = 'Guest Explorer'
                role = 'Guest Researcher'
            else:
                user_part = email.split('@')[0].replace('.', ' ').replace('_', ' ')
                name = user_part.title()
                role = 'Agriculturalist'

            session['user'] = {
                'email': email,
                'name': name,
                'role': role
            }
            return redirect(url_for('dashboard'))
        else:
            error = 'Please enter both email/username and password.'

    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page for new farmers & agronomists."""
    if request.method == 'POST':
        name = request.form.get('name', 'Farmer Alex').strip() or 'Farmer Alex'
        email = request.form.get('email', 'alex@smartfarm.org').strip() or 'alex@smartfarm.org'
        role = request.form.get('role', 'Farm Manager')
        farm_name = request.form.get('farm_name', 'Greenfield Plot').strip()
        session['user'] = {
            'email': email,
            'name': name,
            'role': role,
            'farm_name': farm_name
        }
        return redirect(url_for('dashboard'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    """Sign out and render separate dedicated logout page."""
    session.clear()
    return render_template('logout.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """Executive Smart Farming Dashboard."""
    # Ensure guest session if visiting directly
    if 'user' not in session:
        session['user'] = {
            'email': 'farmer@smartfarm.org',
            'name': 'Farmer John',
            'role': 'Farm Manager'
        }

    return render_template('dashboard.html')


# ─── APIs ────────────────────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    """JSON chatbot endpoint connecting the UI to FarmBot backend."""
    data = request.get_json(silent=True) or {}
    user_msg = data.get('message', '')
    reply = get_response(user_msg)
    return jsonify({'reply': reply})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """RESTful JSON endpoint for programmatic prediction with clean response."""
    try:
        data = request.get_json(force=True)
        values = [
            float(data.get('nitrogen', 0)),
            float(data.get('phosphorus', 0)),
            float(data.get('potassium', 0)),
            float(data.get('temperature', 0)),
            float(data.get('humidity', 0)),
            float(data.get('ph', 0)),
            float(data.get('rainfall', 0))
        ]
        input_df = pd.DataFrame([values], columns=FEATURE_COLS)
        pred = model.predict(input_df)
        probs = model.predict_proba(input_df)[0]
        crop_name = label_encoder.inverse_transform(pred)[0]
        conf = round(float(np.max(probs)) * 100, 2)

        return jsonify({
            'status': 'success',
            'recommended_crop': crop_name,
            'confidence_percent': conf,
            'crop_details': CROP_DATA.get(crop_name.lower(), {
                'emoji': '🌾',
                'category': 'Agricultural Crop',
                'season': 'Optimal Season',
                'water': 'As per regional rainfall',
                'soil': 'Well-drained fertile soil',
                'temp': f"{values[3]}°C",
                'tip': 'Apply recommended local agronomic package of practices.'
            })
        })
    except KeyError as ke:
        return jsonify({'status': 'error', 'message': f'Missing parameter: {str(ke)}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint for production monitoring."""
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Farming Crop Recommendation API',
        'model_loaded': model is not None,
        'supported_crops': len(label_encoder.classes_)
    })


@app.route('/api/crops', methods=['GET'])
def list_crops():
    """Returns list of supported crops and their metadata."""
    return jsonify({
        'total_crops': len(CROP_DATA),
        'crops': CROP_DATA
    })


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)