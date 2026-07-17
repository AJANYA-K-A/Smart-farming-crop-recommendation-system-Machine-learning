import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')          # non-interactive backend – must be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# ─── Load artefacts ───────────────────────────────────────────────────────────
model         = joblib.load('crop_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
df            = pd.read_csv('Crop_recommendation.csv')

# ─── Chart generation ─────────────────────────────────────────────────────────
CHARTS = [
    ("Nitrogen Distribution",        lambda: sns.histplot(df['N'],           kde=True,  color='#4a7c35')),
    ("Phosphorus Distribution",      lambda: sns.histplot(df['P'],           kde=True,  color='#7db65a')),
    ("Potassium Distribution",       lambda: sns.histplot(df['K'],           kde=True,  color='#2d4a1e')),
    ("Temperature Distribution",     lambda: sns.histplot(df['temperature'], kde=True,  color='#e8a020')),
    ("Humidity Distribution",        lambda: sns.histplot(df['humidity'],    kde=True,  color='#3a9abf')),
    ("pH Distribution",              lambda: sns.histplot(df['ph'],          kde=True,  color='#9b5de5')),
    ("Rainfall Distribution",        lambda: sns.histplot(df['rainfall'],    kde=True,  color='#1e88c4')),
    ("Temperature vs Humidity",      lambda: sns.scatterplot(x='temperature', y='humidity',  data=df, alpha=0.5, color='#4a7c35')),
    ("pH vs Rainfall",               lambda: sns.scatterplot(x='ph',         y='rainfall',   data=df, alpha=0.5, color='#e8a020')),
    ("Feature Correlation Heatmap",  lambda: sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='YlGn', fmt='.2f')),
]


def generate_charts():
    """Regenerate all static chart images."""
    sns.set_theme(style='whitegrid', palette='muted')
    for i, (title, plot_fn) in enumerate(CHARTS, 1):
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.sca(ax)
        plot_fn()
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        fig.tight_layout()
        fig.savefig(os.path.join(STATIC_DIR, f'chart{i}.png'), dpi=100)
        plt.close(fig)


# ─── Chatbot helper ───────────────────────────────────────────────────────────
CHAT_RESPONSES = {
    "hello":                           "Hi there! How can I assist you with farming today?",
    "hi":                              "Hello! Need help with crops or irrigation?",
    "what crop should i grow":         "Use the prediction form on the homepage to get a personalised crop suggestion.",
    "which fertilizer is good":        "It depends on your soil type and crop. Common choices are Urea, DAP, and NPK blends.",
    "how to protect crops from pests": "Use integrated pest management (IPM), crop rotation, and targeted pesticide use.",
    "what is smart irrigation":        "Smart irrigation uses sensors and automation to optimise water use in farming.",
    "thank you":                       "You're welcome! Happy farming! 🌾",
    "bye":                             "Goodbye! Feel free to come back anytime.",
}


def get_response(user_input: str) -> str:
    lower = user_input.lower()
    for key, reply in CHAT_RESPONSES.items():
        if key in lower:
            return reply
    return "I'm sorry, I didn't understand that. Please ask something related to farming."


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ''
    form_vals = dict(nitrogen='', phosphorus='', potassium='',
                     temperature='', humidity='', ph='', rainfall='')

    if request.method == 'POST':
        try:
            nitrogen    = float(request.form['nitrogen'])
            phosphorus  = float(request.form['phosphorus'])
            potassium   = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity    = float(request.form['humidity'])
            ph          = float(request.form['ph'])
            rainfall    = float(request.form['rainfall'])

            form_vals = dict(nitrogen=nitrogen, phosphorus=phosphorus,
                             potassium=potassium, temperature=temperature,
                             humidity=humidity, ph=ph, rainfall=rainfall)

            input_data     = np.array([[nitrogen, phosphorus, potassium,
                                        temperature, humidity, ph, rainfall]])
            prediction     = model.predict(input_data)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]
            prediction_text = f'✅ Recommended Crop: {predicted_crop}'

            # Charts are generated only after a successful prediction
            generate_charts()

        except ValueError:
            prediction_text = '⚠️ Please enter valid numbers only.'

    return render_template('index.html', prediction_text=prediction_text, **form_vals)


@app.route('/chat', methods=['POST'])
def chat():
    """Simple JSON endpoint for the chatbot."""
    from flask import jsonify
    user_msg = request.json.get('message', '')
    return jsonify({'reply': get_response(user_msg)})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)