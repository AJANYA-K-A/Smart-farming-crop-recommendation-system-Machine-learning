"""
Smart Farming Crop Recommendation System - Flask Application
Predicts optimal crops based on soil nutrients and climatic parameters.
Features Login Authentication, Executive Dashboard, REST APIs, and FarmBot AI.
"""

import os
import sys
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
CHAT_KNOWLEDGE = [
    {
        'keywords': ['nitrogen', 'urea', 'leaf', 'yellow leaves'],
        'reply': "Nitrogen (N) stimulates rapid leafy growth and chlorophyll synthesis. Nitrogen deficiency causes stunted growth and pale yellowing of older leaves. Supplement with Urea (46% N) or organic manure."
    },
    {
        'keywords': ['phosphorus', 'dap', 'root', 'flowering'],
        'reply': "Phosphorus (P) is crucial for root establishment, energy transfer (ATP), and flower/seed formation. Common sources include Diammonium Phosphate (DAP) and Single Super Phosphate (SSP)."
    },
    {
        'keywords': ['potassium', 'potash', 'mop', 'drought'],
        'reply': "Potassium (K) regulates stomatal opening, strengthens cell walls, improves disease resistance, and boosts drought resilience. Apply Muriate of Potash (MOP) or Sulfate of Potash (SOP)."
    },
    {
        'keywords': ['ph', 'acidic', 'alkaline', 'soil ph', 'lime'],
        'reply': "Most agricultural crops prefer a soil pH between 6.0 and 7.2. For acidic soils (<6.0), add agricultural lime (calcium carbonate). For alkaline soils (>7.5), incorporate gypsum or elemental sulfur."
    },
    {
        'keywords': ['irrigation', 'water', 'drip', 'sprinkler', 'smart irrigation'],
        'reply': "Smart irrigation uses soil moisture sensors and automated timers. Drip irrigation reduces water loss by up to 60% compared to flood irrigation and delivers moisture straight to plant roots."
    },
    {
        'keywords': ['fertilizer', 'npk', 'fertilizer dose', 'compost'],
        'reply': "A balanced N-P-K regimen tailored to your soil test is ideal. Combine inorganic fertilizers with organic compost or vermicompost to maintain microbial soil biodiversity."
    },
    {
        'keywords': ['pest', 'insects', 'ipm', 'disease', 'fungus'],
        'reply': "Adopt Integrated Pest Management (IPM): practice crop rotation, install yellow sticky traps, encourage natural predators (ladybugs), and use neem oil spray before synthetic insecticides."
    },
    {
        'keywords': ['rice', 'paddy'],
        'reply': "Rice thrives in warm, humid climates (20-35°C) with heavy rainfall (150-300 mm) and clayey soil that retains water. Ensure adequate nitrogen and zinc levels."
    },
    {
        'keywords': ['coffee'],
        'reply': "Coffee is ideal in tropical hill slopes (15-28°C) under shade trees, requiring high rainfall (150-200 mm) and porous, organic-rich acidic loam (pH 6.0-6.5)."
    },
    {
        'keywords': ['apple'],
        'reply': "Apples require temperate climates (10-24°C) with 1000+ chilling hours below 7°C during winter, deep well-drained loams, and balanced potassium."
    },
    {
        'keywords': ['watermelon', 'muskmelon', 'melon'],
        'reply': "Melons love warm sunny weather (24-35°C) and well-drained sandy loams. Reduce watering during fruit maturation to maximize sweetness."
    },
    {
        'keywords': ['cotton'],
        'reply': "Cotton requires warm weather (21-35°C), deep black cotton soils (vertisols), and dry sunny conditions during the boll opening stage."
    },
    {
        'keywords': ['crop', 'grow', 'recommend', 'suggest', 'plant'],
        'reply': "You can use the Prediction Form on the homepage! Enter your Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall to get an instant ML-backed recommendation."
    },
    {
        'keywords': ['hello', 'hi', 'hey', 'greetings'],
        'reply': "Hello! I'm FarmBot 🌱, your smart farming advisor. Ask me anything about crops, soil nutrients, fertilizers, pH, or irrigation!"
    },
    {
        'keywords': ['thank', 'thanks', 'appreciate'],
        'reply': "You're very welcome! Wishing you high yields and prosperous farming! 🌾🚜"
    },
    {
        'keywords': ['bye', 'goodbye', 'see you'],
        'reply': "Goodbye! Come back whenever you need smart agricultural advice."
    }
]


def get_response(user_input: str) -> str:
    """Intelligent rule-based agricultural assistant reply."""
    if not user_input or not user_input.strip():
        return "Please ask a question regarding crops, soil nutrients (NPK), climate, pH, or irrigation!"

    lower = user_input.lower()
    for entry in CHAT_KNOWLEDGE:
        for kw in entry['keywords']:
            if kw in lower:
                return entry['reply']

    return (
        "That's an interesting farming question! For specific crop suggestions, try our prediction form above. "
        "You can also ask me about Nitrogen, Phosphorus, Potassium, Soil pH, Irrigation, or specific crops like Rice, Coffee, and Apples."
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


@app.route('/logout')
def logout():
    """Sign out and return to login."""
    session.clear()
    return redirect(url_for('login'))


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